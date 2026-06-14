# Model Card: BayLing-Duplex

## Model Summary

BayLing-Duplex is a native full-duplex speech language model. It receives continuous user speech and autoregressively predicts assistant dialogue-state tokens, assistant text tokens, and assistant speech tokens in a multi-channel interleaved sequence.

## Intended Use

- Research on full-duplex spoken dialogue.
- Turn-taking and interruption modeling.
- Speech-to-speech assistant prototypes.

This model is not intended for deployment in safety-critical or consent-sensitive environments without additional safeguards.

## Architecture

- Backbone: 9B decoder-only Transformer.
- Speech tokenizer: Whisper-style VQ encoder, 12.5 Hz token rate.
- Speech decoder: flow-matching acoustic decoder + HiFT vocoder.
- Inference: block-by-block autoregressive decoding with text-token and speech-token masks.
- Default block ratio: `10:5:10`.

## Training Data

The paper uses 400K synthesized full-duplex samples:

- 200K turn-taking samples.
- 200K interruption samples.

The data is generated from multi-turn speech-to-speech dialogues and converted into aligned user-speech, assistant-text, and assistant-speech channels.

## Metrics

Reported in the paper:

- Turn-taking SR@3s: 92.0.
- Interruption success ISR@2s: 100.0.
- Interruption overlap: 1.10 s.
- Llama Questions accuracy: 46.0.
- Web Questions accuracy: 18.1.

## Limitations

- Evaluated primarily on synthetic near-field speech.
- May not be robust to noisy or multi-speaker audio.
- Not evaluated for backchannels, multi-party dialogue, or emotion-aware turn-taking.
- Inherits biases and limitations from its base speech-language stack.

## Safety and Misuse

The model can produce natural speech and may be misused for impersonation, social engineering, or audio misinformation. Deployments should include consent, speaker verification, watermarking, and abuse monitoring.

## License

See `LICENSE` and `NOTICE.md`.

## Acknowledgements

BayLing-Duplex is trained based on GLM-4-Voice and uses components released by the GLM-4-Voice team. We thank the GLM-4-Voice team for making their model and code available to the community.
