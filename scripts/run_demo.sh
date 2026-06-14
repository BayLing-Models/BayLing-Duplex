#!/usr/bin/env bash
set -euo pipefail

python -m bayling_duplex.cli \
  --model-path models/bayling_duplex_model \
  --speech-tokenizer-path models/speech_tokenizer \
  --decoder-path models/speech_decoder \
  --input-audio examples/input.wav \
  --output-json outputs/result.json \
  --output-audio outputs/response.wav \
  --interleave-ratio 10:5:10 \
  --max-duration 60 \
  --temperature 0.6 \
  --top-p 0.8
