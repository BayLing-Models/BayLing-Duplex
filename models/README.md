# Model Directory

Place model files here or pass external paths to the CLI.

Download the expected local layout with:

```bash
mkdir -p models
hf download BayLing-Models/BayLing-Duplex \
  --repo-type model \
  --local-dir models/bayling_duplex_model
hf download zai-org/glm-4-voice-tokenizer \
  --repo-type model \
  --local-dir models/speech_tokenizer
hf download zai-org/glm-4-voice-decoder \
  --repo-type model \
  --local-dir models/speech_decoder
```

Expected layout after download:

```text
models/
  bayling_duplex_model/
  speech_tokenizer/
  speech_decoder/
```

Large files such as `*.safetensors`, `*.pt`, and `*.bin` should be tracked with Git LFS.
