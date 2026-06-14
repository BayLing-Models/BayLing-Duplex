import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torchaudio
from transformers import AutoModel, AutoTokenizer, WhisperFeatureExtractor


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_BAYLING_RUNTIME_SRC = _PACKAGE_ROOT / "bayling_duplex_runtime"
_MATCHA_TTS_SRC = _BAYLING_RUNTIME_SRC / "third_party" / "Matcha-TTS"

for _path in (_MATCHA_TTS_SRC, _BAYLING_RUNTIME_SRC, _PACKAGE_ROOT):
    _path_str = str(_path)
    while _path_str in sys.path:
        sys.path.remove(_path_str)
    sys.path.insert(0, _path_str)

from bayling_duplex_runtime.flow_inference import AudioDecoder  # noqa: E402
from bayling_duplex_runtime.speech_tokenizer.modeling_whisper import WhisperVQEncoder  # noqa: E402


AudioInput = Union[str, os.PathLike, torch.Tensor, Tuple[torch.Tensor, int]]


@dataclass
class ResponseSegment:
    text: str
    text_token_start: int
    text_token_end: Optional[int]
    start_block: int
    end_block: Optional[int]
    start_time: float
    turn_taking_time: Optional[float]
    audio_tokens: List[int] = field(default_factory=list)

    def to_dict(self, include_tokens: bool = False) -> Dict[str, Any]:
        data = {
            "text": self.text,
            "text_token_start": self.text_token_start,
            "text_token_end": self.text_token_end,
            "start_block": self.start_block,
            "end_block": self.end_block,
            "start_time": self.start_time,
            "turn_taking_time": self.turn_taking_time,
            "audio_token_count": len(self.audio_tokens),
        }
        if include_tokens:
            data["audio_tokens"] = self.audio_tokens
        return data


@dataclass
class DuplexResult:
    text: str
    text_tokens: List[int]
    audio_tokens: List[int]
    user_audio_tokens: List[int]
    response_audio_tokens: List[int]
    segments: List[ResponseSegment]
    total_blocks: int
    found_assistant: bool
    found_epad: bool
    reached_max_blocks: bool
    user_audio_duration: Optional[float]
    interleave_ratio: str

    def to_dict(self, include_tokens: bool = False) -> Dict[str, Any]:
        data = {
            "text": self.text,
            "total_blocks": self.total_blocks,
            "found_assistant": self.found_assistant,
            "found_epad": self.found_epad,
            "reached_max_blocks": self.reached_max_blocks,
            "user_audio_duration": self.user_audio_duration,
            "interleave_ratio": self.interleave_ratio,
            "text_token_count": len(self.text_tokens),
            "audio_token_count": len(self.audio_tokens),
            "user_audio_token_count": len(self.user_audio_tokens),
            "response_audio_token_count": len(self.response_audio_tokens),
            "segments": [segment.to_dict(include_tokens=include_tokens) for segment in self.segments],
        }
        if include_tokens:
            data.update(
                {
                    "text_tokens": self.text_tokens,
                    "audio_tokens": self.audio_tokens,
                    "user_audio_tokens": self.user_audio_tokens,
                    "response_audio_tokens": self.response_audio_tokens,
                }
            )
        return data

    def save_json(self, path: Union[str, os.PathLike], include_tokens: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(include_tokens=include_tokens), f, indent=2, ensure_ascii=False)


def _patch_model_for_new_transformers(model: torch.nn.Module) -> None:
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        return

    original_forward = model.forward

    def has_real_data(past_key_values: Any) -> bool:
        try:
            return len(past_key_values) > 0 and bool(past_key_values.key_cache)
        except (AttributeError, TypeError):
            return False

    def patched_forward(*args: Any, **kwargs: Any) -> Any:
        past_key_values = kwargs.get("past_key_values")
        if isinstance(past_key_values, DynamicCache):
            kwargs["past_key_values"] = (
                past_key_values.to_legacy_cache() if has_real_data(past_key_values) else None
            )
        return original_forward(*args, **kwargs)

    model.forward = patched_forward

    if not hasattr(model, "_extract_past_from_model_output"):

        def extract_past_from_model_output(self: Any, outputs: Any, **_: Any) -> Tuple[str, Any]:
            return "past_key_values", getattr(outputs, "past_key_values", None)

        model._extract_past_from_model_output = extract_past_from_model_output.__get__(model)


class BayLingDuplex:
    sample_rate: int = 16000
    output_sample_rate: int = 22050

    def __init__(
        self,
        model_path: Union[str, os.PathLike],
        speech_tokenizer_path: Union[str, os.PathLike],
        decoder_path: Optional[Union[str, os.PathLike]] = None,
        interleave_ratio: str = "10:5:10",
        device: Optional[str] = None,
        torch_dtype: Optional[str] = "auto",
        audio_vocab_size: int = 16384,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.audio_vocab_size = audio_vocab_size
        self.x_ratio, self.y_ratio, self.z_ratio = self._parse_interleave_ratio(interleave_ratio)
        self.interleave_ratio = f"{self.x_ratio}:{self.y_ratio}:{self.z_ratio}"
        self.time_offset = self.x_ratio / 12.5
        self.audio_to_text_ratio = self.x_ratio / self.y_ratio

        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if torch_dtype:
            model_kwargs["torch_dtype"] = self._resolve_torch_dtype(torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, **model_kwargs).to(self.device).eval()
        _patch_model_for_new_transformers(self.model)

        self.speech_tokenizer = WhisperVQEncoder.from_pretrained(speech_tokenizer_path).eval().to(self.device)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(speech_tokenizer_path)

        self.decoder: Optional[AudioDecoder] = None
        if decoder_path is not None:
            decoder_path = Path(decoder_path)
            self.decoder = AudioDecoder(
                config_path=str(decoder_path / "config.yaml"),
                flow_ckpt_path=str(decoder_path / "flow.pt"),
                hift_ckpt_path=str(decoder_path / "hift.pt"),
                device=str(self.device),
            )

        self._cache_special_tokens()

    @staticmethod
    def _resolve_torch_dtype(torch_dtype: str) -> Union[str, torch.dtype]:
        if torch_dtype == "auto":
            return "auto"
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if torch_dtype not in mapping:
            raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
        return mapping[torch_dtype]

    @staticmethod
    def _parse_interleave_ratio(ratio: str) -> Tuple[int, int, int]:
        parts = [int(part.strip()) for part in ratio.split(":")]
        if len(parts) == 2:
            x_ratio, z_ratio = parts
            y_ratio = 1
        elif len(parts) == 3:
            x_ratio, y_ratio, z_ratio = parts
        else:
            raise ValueError(f"Invalid interleave ratio: {ratio}")
        if x_ratio <= 0 or y_ratio <= 0 or z_ratio <= 0:
            raise ValueError(f"Interleave ratio must be positive: {ratio}")
        if x_ratio != z_ratio:
            raise ValueError("This inference path expects user and assistant audio ratios to match.")
        return x_ratio, y_ratio, z_ratio

    def _optional_token_id(self, token: str) -> Optional[int]:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        if token_id is None or (unk_id is not None and token_id == unk_id and token != self.tokenizer.unk_token):
            return None
        return int(token_id)

    def _required_token_id(self, token: str) -> int:
        token_id = self._optional_token_id(token)
        if token_id is None:
            raise ValueError(f"Required token is missing from tokenizer: {token}")
        return token_id

    def _cache_special_tokens(self) -> None:
        self.audio_token_start = self._required_token_id("<|audio_0|>")
        self.audio_token_end = self.audio_token_start + self.audio_vocab_size
        self.end_token_id = self._optional_token_id("<|endoftext|>")
        self.user_token_id = self._optional_token_id("<|user|>")
        self.assistant_token_id = self._required_token_id("<|assistant|>")
        self.silence_token_id = self._optional_token_id("[SILENCE]")
        self.pad_token_id = self._optional_token_id("[PAD]")
        self.epad_token_id = self._required_token_id("[EPAD]")
        self._text_special_token_ids = {
            token_id
            for token_id in (
                self.silence_token_id,
                self.pad_token_id,
                self.epad_token_id,
                self.user_token_id,
                self.assistant_token_id,
                self.end_token_id,
            )
            if token_id is not None
        }

    def load_audio(self, audio: AudioInput) -> Tuple[torch.Tensor, float]:
        if isinstance(audio, tuple):
            waveform, sample_rate = audio
        elif isinstance(audio, torch.Tensor):
            waveform, sample_rate = audio, self.sample_rate
        else:
            waveform, sample_rate = torchaudio.load(str(audio))

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        duration = waveform.shape[1] / self.sample_rate
        return waveform.contiguous(), duration

    def pad_or_trim_audio(self, waveform: torch.Tensor, target_duration: float) -> torch.Tensor:
        target_samples = int(target_duration * self.sample_rate)
        current_samples = waveform.shape[1]
        if current_samples < target_samples:
            pad = torch.zeros(waveform.shape[0], target_samples - current_samples, dtype=waveform.dtype)
            waveform = torch.cat([waveform, pad], dim=1)
        elif current_samples > target_samples:
            waveform = waveform[:, :target_samples]
        return waveform

    @torch.no_grad()
    def tokenize_audio(self, audio: AudioInput) -> Tuple[List[int], float]:
        waveform, duration = self.load_audio(audio)
        audio_np = waveform[0].cpu().numpy()
        all_tokens: List[int] = []

        pooling_kernel_size = self.speech_tokenizer.config.pooling_kernel_size or 1
        stride = (
            self.speech_tokenizer.conv1.stride[0]
            * self.speech_tokenizer.conv2.stride[0]
            * pooling_kernel_size
            * self.feature_extractor.hop_length
        )

        time_step = 0
        while time_step * self.sample_rate < len(audio_np):
            audio_segment = audio_np[time_step * self.sample_rate : (time_step + 30) * self.sample_rate]
            features = self.feature_extractor(
                [audio_segment],
                sampling_rate=self.sample_rate,
                return_attention_mask=True,
                return_tensors="pt",
                padding="longest",
                pad_to_multiple_of=stride,
            ).to(self.device)

            outputs = self.speech_tokenizer(**features)
            speech_tokens = outputs.quantized_token_ids
            attention_mask = features.attention_mask[
                :, :: self.speech_tokenizer.conv1.stride[0] * self.speech_tokenizer.conv2.stride[0]
            ]
            attention_mask = attention_mask[:, :: pooling_kernel_size]
            all_tokens.extend(speech_tokens[0][attention_mask[0].bool()].tolist())
            time_step += 30

        return all_tokens, duration

    def _mask_logits(self, logits: torch.Tensor, mode: str) -> torch.Tensor:
        masked_logits = logits.clone()
        audio_start = min(self.audio_token_start, masked_logits.shape[-1])
        audio_end = min(self.audio_token_end, masked_logits.shape[-1])

        if mode == "text":
            masked_logits[:, audio_start:audio_end] = float("-inf")
            return masked_logits

        if mode == "audio":
            mask = torch.ones(masked_logits.shape[-1], dtype=torch.bool, device=masked_logits.device)
            mask[audio_start:audio_end] = False
            masked_logits[:, mask] = float("-inf")
            return masked_logits

        return masked_logits

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        mode: str,
        silence_penalty: float = 0.0,
    ) -> int:
        logits = self._mask_logits(logits, mode)

        if silence_penalty > 0 and self.silence_token_id is not None and mode == "text":
            logits[:, self.silence_token_id] -= silence_penalty

        if temperature <= 0:
            return int(torch.argmax(logits, dim=-1)[0].item())

        logits = logits / temperature
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1)[0, 0].item())

    def _should_apply_silence_penalty(self, assistant_count: int, epad_count: int, max_epad_count: int) -> bool:
        if epad_count >= max_epad_count:
            return False
        return assistant_count <= epad_count

    @torch.no_grad()
    def generate_from_audio_tokens(
        self,
        user_audio_tokens: Sequence[int],
        user_audio_duration: Optional[float],
        max_duration: float = 60.0,
        temperature: float = 0.6,
        top_p: float = 0.8,
        silence_penalty: float = 0.0,
        max_epad_count: int = 1,
    ) -> DuplexResult:
        num_tokens = (len(user_audio_tokens) // self.x_ratio) * self.x_ratio
        user_audio_tokens = list(user_audio_tokens[:num_tokens])
        max_blocks = int(max_duration * 12.5) // self.x_ratio
        num_available_blocks = num_tokens // self.x_ratio
        block_count = min(num_available_blocks, max_blocks)

        user_token_ids = [token + self.audio_token_start for token in user_audio_tokens]
        text_tokens: List[int] = []
        audio_tokens: List[int] = []
        past_key_values = None
        current_position = 0
        assistant_count = 0
        epad_count = 0
        stop_after_block = False
        block_idx = -1

        for block_idx in range(block_count):
            start = block_idx * self.x_ratio
            block_user_tokens = user_token_ids[start : start + self.x_ratio]
            current_input = torch.tensor([block_user_tokens], device=self.device)
            position_ids = torch.arange(
                current_position,
                current_position + len(block_user_tokens),
                device=self.device,
            ).unsqueeze(0)
            outputs = self.model(
                input_ids=current_input,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            current_position += len(block_user_tokens)

            for _ in range(self.y_ratio):
                apply_penalty = self._should_apply_silence_penalty(assistant_count, epad_count, max_epad_count)
                token_id = self._sample_token(
                    logits,
                    temperature=temperature,
                    top_p=top_p,
                    mode="text",
                    silence_penalty=silence_penalty if apply_penalty else 0.0,
                )
                text_tokens.append(token_id)

                if token_id == self.assistant_token_id:
                    assistant_count += 1
                elif token_id == self.epad_token_id:
                    epad_count += 1
                    if epad_count >= max_epad_count:
                        stop_after_block = True
                elif self.end_token_id is not None and token_id == self.end_token_id:
                    stop_after_block = True

                current_input = torch.tensor([[token_id]], device=self.device)
                position_ids = torch.tensor([[current_position]], device=self.device)
                outputs = self.model(
                    input_ids=current_input,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                current_position += 1

            for _ in range(self.z_ratio):
                token_id = self._sample_token(logits, temperature=temperature, top_p=top_p, mode="audio")
                audio_tokens.append(token_id - self.audio_token_start)

                current_input = torch.tensor([[token_id]], device=self.device)
                position_ids = torch.tensor([[current_position]], device=self.device)
                outputs = self.model(
                    input_ids=current_input,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                current_position += 1

            if stop_after_block:
                break

        total_blocks = block_idx + 1 if block_idx >= 0 else 0
        processed_user_tokens = user_audio_tokens[: total_blocks * self.x_ratio]
        segments = self._extract_segments(text_tokens, audio_tokens, user_audio_duration)
        first_segment = segments[0] if segments else None
        response_audio_tokens = first_segment.audio_tokens if first_segment else []
        text = first_segment.text if first_segment else self._decode_text(text_tokens)

        return DuplexResult(
            text=text,
            text_tokens=text_tokens,
            audio_tokens=audio_tokens,
            user_audio_tokens=processed_user_tokens,
            response_audio_tokens=response_audio_tokens,
            segments=segments,
            total_blocks=total_blocks,
            found_assistant=assistant_count > 0,
            found_epad=epad_count > 0,
            reached_max_blocks=total_blocks >= max_blocks and not stop_after_block,
            user_audio_duration=user_audio_duration,
            interleave_ratio=self.interleave_ratio,
        )

    @torch.no_grad()
    def generate(
        self,
        audio: AudioInput,
        max_duration: float = 60.0,
        temperature: float = 0.6,
        top_p: float = 0.8,
        silence_penalty: float = 0.0,
        max_epad_count: int = 1,
        pad_to_max_duration: bool = True,
    ) -> DuplexResult:
        waveform, user_duration = self.load_audio(audio)
        model_audio = self.pad_or_trim_audio(waveform, max_duration) if pad_to_max_duration else waveform
        user_audio_tokens, _ = self.tokenize_audio((model_audio, self.sample_rate))
        return self.generate_from_audio_tokens(
            user_audio_tokens=user_audio_tokens,
            user_audio_duration=user_duration,
            max_duration=max_duration,
            temperature=temperature,
            top_p=top_p,
            silence_penalty=silence_penalty,
            max_epad_count=max_epad_count,
        )

    def _decode_text(self, token_ids: Sequence[int]) -> str:
        text_ids = [token_id for token_id in token_ids if token_id not in self._text_special_token_ids]
        if not text_ids:
            return ""
        return self.tokenizer.decode(text_ids, skip_special_tokens=False).strip()

    def _extract_segments(
        self,
        text_tokens: Sequence[int],
        audio_tokens: Sequence[int],
        user_audio_duration: Optional[float],
    ) -> List[ResponseSegment]:
        segments: List[ResponseSegment] = []
        current_start: Optional[int] = None

        for text_idx, token_id in enumerate(text_tokens):
            if token_id == self.assistant_token_id:
                current_start = text_idx
                continue
            if token_id == self.epad_token_id and current_start is not None:
                segments.append(self._build_segment(current_start, text_idx, text_tokens, audio_tokens, user_audio_duration))
                current_start = None

        if current_start is not None:
            segments.append(self._build_segment(current_start, None, text_tokens, audio_tokens, user_audio_duration))

        return segments

    def _build_segment(
        self,
        start_text_idx: int,
        end_text_idx: Optional[int],
        text_tokens: Sequence[int],
        audio_tokens: Sequence[int],
        user_audio_duration: Optional[float],
    ) -> ResponseSegment:
        start_block = start_text_idx // self.y_ratio
        end_block = (end_text_idx // self.y_ratio) if end_text_idx is not None else None
        audio_start = start_block * self.z_ratio
        audio_end = ((end_block + 1) * self.z_ratio) if end_block is not None else len(audio_tokens)
        segment_audio_tokens = list(audio_tokens[audio_start:audio_end])
        text_end = end_text_idx if end_text_idx is not None else len(text_tokens)
        text = self._decode_text(text_tokens[start_text_idx + 1 : text_end])
        start_time = start_text_idx * self.audio_to_text_ratio / 12.5 + self.time_offset
        turn_taking_time = None
        if user_audio_duration is not None:
            turn_taking_time = start_time - user_audio_duration

        return ResponseSegment(
            text=text,
            text_token_start=start_text_idx,
            text_token_end=end_text_idx,
            start_block=start_block,
            end_block=end_block,
            start_time=start_time,
            turn_taking_time=turn_taking_time,
            audio_tokens=segment_audio_tokens,
        )

    @torch.no_grad()
    def tokens_to_audio(self, audio_tokens: Sequence[int]) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError("Audio decoder is not loaded. Pass decoder_path when constructing the model.")
        if not audio_tokens:
            return torch.zeros(1, self.output_sample_rate)

        prompt_speech_feat = torch.zeros(1, 0, 80, device=self.device)
        prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64, device=self.device)
        tts_token = torch.tensor(list(audio_tokens), dtype=torch.int64, device=self.device).unsqueeze(0)
        speech, _ = self.decoder.token2wav(
            tts_token,
            uuid=str(uuid.uuid4()),
            prompt_token=prompt_speech_token,
            prompt_feat=prompt_speech_feat,
            finalize=True,
        )
        return speech.cpu()

    def save_audio(
        self,
        audio_tokens: Sequence[int],
        path: Union[str, os.PathLike],
        sample_rate: Optional[int] = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        waveform = self.tokens_to_audio(audio_tokens)
        torchaudio.save(str(path), waveform, sample_rate or self.output_sample_rate, format="wav")
