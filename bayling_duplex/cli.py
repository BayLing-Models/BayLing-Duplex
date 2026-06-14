import argparse
from pathlib import Path

from .duplex import BayLingDuplex


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal BayLingDuplex full-duplex inference")
    parser.add_argument("--model-path", required=True, help="Path to the fine-tuned BayLingDuplex model")
    parser.add_argument("--speech-tokenizer-path", required=True, help="Path to the speech tokenizer")
    parser.add_argument("--decoder-path", default=None, help="Path to decoder directory with config.yaml, flow.pt, hift.pt")
    parser.add_argument("--input-audio", required=True, help="Input user wav/flac/mp3 file")
    parser.add_argument("--output-json", default="outputs/result.json", help="Path to write metadata JSON")
    parser.add_argument("--output-audio", default="outputs/response.wav", help="Path to write decoded assistant wav")
    parser.add_argument("--interleave-ratio", default="10:5:10", help="full-duplex ratio, e.g. 10:5:10")
    parser.add_argument("--max-duration", type=float, default=60.0, help="Max user-channel duration after padding")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--silence-penalty", type=float, default=0.0)
    parser.add_argument("--max-epad-count", type=int, default=1, help="Use 2 for simple interruption / two-turn input")
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu. Defaults to cuda when available")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--include-tokens", action="store_true", help="Include raw token arrays in JSON")
    parser.add_argument(
        "--synthesize",
        choices=["response", "all"],
        default="response",
        help="Decode the first assistant segment or the full assistant audio channel",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip wav decoding. Useful when decoder weights are not available.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.no_audio and args.decoder_path is None:
        raise ValueError("--decoder-path is required unless --no-audio is set")

    model = BayLingDuplex(
        model_path=args.model_path,
        speech_tokenizer_path=args.speech_tokenizer_path,
        decoder_path=None if args.no_audio else args.decoder_path,
        interleave_ratio=args.interleave_ratio,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    result = model.generate(
        audio=args.input_audio,
        max_duration=args.max_duration,
        temperature=args.temperature,
        top_p=args.top_p,
        silence_penalty=args.silence_penalty,
        max_epad_count=args.max_epad_count,
    )
    result.save_json(args.output_json, include_tokens=args.include_tokens)

    audio_path = None
    if not args.no_audio:
        tokens = result.audio_tokens if args.synthesize == "all" else result.response_audio_tokens
        model.save_audio(tokens, args.output_audio)
        audio_path = str(Path(args.output_audio))

    print(f"text: {result.text}")
    print(f"json: {Path(args.output_json)}")
    if audio_path:
        print(f"audio: {audio_path}")


if __name__ == "__main__":
    main()
