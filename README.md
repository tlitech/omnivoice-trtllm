# OmniVoice on Modal

Deploy OmniVoice TTS model using Triton Inference Server on Modal.

Supports two inference backends:
- **PyTorch** — zero setup, works out of the box
- **TRT-LLM** — faster inference via TensorRT-LLM engine for the Qwen3 backbone

OmniVoice supports 600+ languages with three synthesis modes:
- **Voice Clone** — clone any voice from a reference audio
- **Voice Design** — create custom voices via text instructions (gender, age, accent, etc.)
- **Auto Voice** — model picks a voice automatically

## Prerequisites

- Modal CLI: `pip install modal`
- Modal auth: `modal setup`
- L4 GPU quota

## Quick Start (TRT-LLM)

Run the complete pipeline:

```bash
modal run omnivoice_trtllm_modal.py
```

This will:
1. Download OmniVoice model from HuggingFace
2. Convert checkpoint and build TRT-LLM engine
3. Start Triton server and run test inference on voice samples

## Deploy

```bash
modal deploy omnivoice_trtllm_modal.py
```

## Test

`--server-url` is required for all commands below. Use the URL from `modal deploy` output.

### Run all voice samples

```bash
python test_client_http.py \
  --server-url https://<your-app>--omnivoice-triton-server-serve.modal.run \
  --voices-dir voices/system
```

Runs all 17 samples, saves output to `voices/output/`, logs ref text, target text, latency per sample.

`--reference-text` is optional — if omitted, the server auto-transcribes via Whisper.

### Voice Clone — Vietnamese (8 samples)

Clone a voice from reference audio. The model preserves the speaker's voice characteristics.

```bash
# Speaker 1 (fleurs dataset)
python test_client_http.py \
  --server-url https://<your-app>--omnivoice-triton-server-serve.modal.run \
  --reference-audio voices/system/vi_01.wav \
  --reference-text "Họ thường cung cấp những thức ăn, đồ uống và giải trí đặc biệt để giữ khách hàng thoải mái, và giữ họ ở lại." \
  --target-text "Lực ma sát trên lòng đường đóng băng và phủ tuyết thấp khiến bạn không thể lái xe như đang lái trên đường nhựa bình thường."

# Speaker 2 (minimax dataset)
python test_client_http.py \
  --server-url https://<your-app>--omnivoice-triton-server-serve.modal.run \
  --reference-audio voices/system/vi_05.wav \
  --reference-text "Hà sợ quá, giật lùi lại phía sau, xuyết ngã ngựa ra đất" \
  --target-text "Sách nói là phương tiện tuyệt vời để khám phá những câu chuyện hay trong cuộc sống."
```

All Vietnamese samples:

| Sample | Ref text | Target text |
|--------|----------|-------------|
| vi_01 | Họ thường cung cấp những thức ăn... | Lực ma sát trên lòng đường đóng băng... |
| vi_02 | Hiển nhiên, nếu bạn biết một ngôn ngữ La Mã... | Du ngoạn bằng đường thủy trên đất liền... |
| vi_03 | Trong khoảng 150 đến 200 bản in... | Với đặc điểm tương đối khó tiếp cận, Timbuktu... |
| vi_04 | Quyền lực toàn diện của nó... | Nó cũng không có thẩm quyền trong việc sửa đổi... |
| vi_05 | Hà sợ quá, giật lùi lại phía sau... | Sách nói là phương tiện tuyệt vời... |
| vi_06 | Hà sợ quá, giật lùi lại phía sau... | Bản tin này giúp tôi hiểu rõ hơn... |
| vi_07 | Trinh có hỏi thì được biết sông Trai... | Vận động viên Paralympic đã giành huy chương vàng... |
| vi_08 | Trinh có hỏi thì được biết sông Trai... | Tôi đang tìm một căn hộ mới gần nơi làm việc... |

### Cross-Lingual Zero-shot TTS (3 samples)

Clone a voice and speak in a different language. Chinese voice speaks English, English voice speaks Chinese.

```bash
# Chinese voice -> English output
python test_client_http.py \
  --server-url https://<your-app>--omnivoice-triton-server-serve.modal.run \
  --reference-audio voices/system/cross_01_zh.wav \
  --reference-text "对，这就是我，万人敬仰的太乙真人，虽然有点婴儿肥，但也掩不住我逼人的帅气。" \
  --target-text 'Suddenly, there was a burst of laughter beside me. I looked at them, stood up straight with high spirit.'

# English voice -> Chinese output
python test_client_http.py \
  --server-url https://<your-app>--omnivoice-triton-server-serve.modal.run \
  --reference-audio voices/system/cross_02_en.wav \
  --reference-text "Some call me nature. Others call me Mother Nature." \
  --target-text "顿时，气氛变得沉郁起来。乍看之下，一切的困扰仿佛都围绕在我身边。"
```

| Sample | Ref language | Target language | Ref text | Target text |
|--------|-------------|-----------------|----------|-------------|
| cross_01_zh | Chinese | English | 对，这就是我，万人敬仰的太乙真人... | Suddenly, there was a burst of laughter... |
| cross_02_en | English | Chinese | Some call me nature. Others call me Mother Nature... | 顿时，气氛变得沉郁起来... |
| cross_03_en | English | Chinese | Are you familiar with it? Slice the steak... | 我抬起头，坚定地说："身高不能决定一切..." |

### Voice Design (3 samples)

Create a voice from text description alone — no reference audio needed.

```bash
python test_client_http.py \
  --server-url https://<your-app>--omnivoice-triton-server-serve.modal.run \
  --instruct "Female, Child" \
  --target-text "The world is full of amazing wonders. There are high mountains, blue oceans, and lovely animals everywhere."

python test_client_http.py \
  --server-url https://<your-app>--omnivoice-triton-server-serve.modal.run \
  --instruct "Male, High Pitch, Indian Accent" \
  --target-text "The world is full of amazing wonders. There are high mountains, blue oceans, and lovely animals everywhere."

python test_client_http.py \
  --server-url https://<your-app>--omnivoice-triton-server-serve.modal.run \
  --instruct "Female, Elderly, British Accent" \
  --target-text "The world is full of amazing wonders. There are high mountains, blue oceans, and lovely animals everywhere."
```

| Sample | Instruct | Target text | Status |
|--------|----------|-------------|--------|
| design_01 | Female, Child | The world is full of amazing wonders... | OK |
| design_02 | Male, High Pitch, Indian Accent | The world is full of amazing wonders... | Known issue: noisy output |
| design_03 | Female, Elderly, British Accent | The world is full of amazing wonders... | OK |

> **Note**: `design_02` produces noisy output. This is likely caused by the TRT-LLM checkpoint conversion (PyTorch mode produces clean output). Needs investigation.

### Fine-Grained Control (3 samples)

Control speech with special tokens — emotions, pronunciations. No reference audio needed.

```bash
# Laughter
python test_client_http.py \
  --server-url https://<your-app>--omnivoice-triton-server-serve.modal.run \
  --target-text "[laughter] You really got me. I didn't see that coming at all."

# Mixed emotions (Chinese)
python test_client_http.py \
  --server-url https://<your-app>--omnivoice-triton-server-serve.modal.run \
  --target-text "[dissatisfaction-hnn]这个结果我不太满意。[surprise-oh]原来你有备用方案？[laughter]那太好了。"

# Phonetic control (ARPAbet)
python test_client_http.py \
  --server-url https://<your-app>--omnivoice-triton-server-serve.modal.run \
  --target-text "He plays the [B EY1 S] guitar while catching a [B AE1 S] fish."
```

| Sample | Target text | Control type |
|--------|-------------|--------------|
| fine_01 | [laughter] You really got me... | Emotion token |
| fine_02 | [dissatisfaction-hnn]这个结果我不太满意... | Mixed emotions |
| fine_03 | He plays the [B EY1 S] guitar... | ARPAbet pronunciation |

### Benchmark (warm-up + N runs)

```bash
python test_client_http.py \
  --server-url https://<your-app>--omnivoice-triton-server-serve.modal.run \
  --reference-audio voices/system/vi_01.wav \
  --reference-text "Họ thường cung cấp những thức ăn, đồ uống và giải trí đặc biệt để giữ khách hàng thoải mái, và giữ họ ở lại." \
  --target-text "Lực ma sát trên lòng đường đóng băng và phủ tuyết thấp khiến bạn không thể lái xe như đang lái trên đường nhựa bình thường." \
  --loop 10
```

## API

### Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `target_text` | STRING | Yes | Text to synthesize |
| `reference_wav` | FP32 | No | Reference audio waveform (24kHz, mono) |
| `reference_wav_len` | INT32 | No | Length of reference audio in samples |
| `reference_text` | STRING | No | Transcript of reference audio |
| `language` | STRING | No | Language name (`English`) or code (`en`) |
| `instruct` | STRING | No | Voice design instruction |

### Output

| Name | Type | Description |
|------|------|-------------|
| `waveform` | FP32 | Generated audio waveform (24kHz) |

## Architecture

### TRT-LLM mode (default)

OmniVoice loads the full PyTorch model, then **monkey-patches `llm.forward`** to route the Qwen3 transformer through a TRT-LLM engine. All other logic (embeddings, generation loop, audio codec) stays untouched in the original OmniVoice code.

```
OmniVoice.generate()
  -> _generate_iterative()                              [original code]
    -> _prepare_embed_inputs()
        -> llm.model.embed_tokens (text embedding)      [PyTorch, from LLM object]
        -> audio_embeddings (multi-codebook)             [PyTorch]
    -> self.llm(inputs_embeds=...)
        -> llm.forward (monkey-patched)                  [TRT-LLM engine ← replaces Qwen3]
    -> audio_heads (logits)                              [PyTorch]
    -> CFG + token selection                             [original code]
  -> audio_tokenizer.decode()                            [PyTorch]
```

Only the Qwen3 transformer backbone (28 layers, ~840MB) runs on TRT-LLM.
Everything else uses the original OmniVoice PyTorch code, unchanged.

### What TRT-LLM replaces

| Component | PyTorch mode | TRT-LLM mode |
|-----------|-------------|---------------|
| Text embedding (`llm.model.embed_tokens`) | PyTorch | PyTorch (from LLM object) |
| **Qwen3 transformer (28 layers)** | **PyTorch** | **TRT-LLM engine** |
| Audio embeddings | PyTorch | PyTorch |
| Audio heads | PyTorch | PyTorch |
| Generation loop (iterative unmasking) | OmniVoice code | OmniVoice code (same) |
| Audio codec (HiggsAudioV2) | PyTorch | PyTorch |
| Text tokenizer | HuggingFace | HuggingFace |

### Build pipeline

```
HF checkpoint (k2-fsa/OmniVoice)
  -> convert_checkpoint.py    (extract Qwen3 layer weights to TRT-LLM format)
  -> patch/omnivoice/         (TRT-LLM model definition: attention + MLP + RoPE)
  -> trtllm-build             (compile TRT engine)
  -> rank0.engine             (~840MB, FP16)
```

### PyTorch mode (fallback)

Pass `use_trtllm=False` to `start_and_test_triton_server`, or set `trtllm` to empty in config template. No engine build needed.

### Debug mode

Pass `debug=True` to `start_and_test_triton_server` to:
- Validate TRT engine output vs PyTorch (cosine similarity check)
- Print engine I/O stats on first forward pass


## Benchmark

### Results (NVIDIA L4, 10 voice samples, FP16)

| Metric | TRT-LLM | PyTorch | Speedup |
|--------|---------|---------|---------|
| Mean latency (e2e) | 2.47s | 2.97s | 1.20x |
| Min latency (e2e) | 1.89s | 2.52s | 1.33x |
| Max latency (e2e) | 2.83s | 3.53s | 1.25x |
| Mean server infer | ~1.22s | ~1.67s | 1.37x |

- **Test set**: 10 samples — 4 Vietnamese, 3 cross-lingual, 3 fine-grained control
- **Audio output**: ~7-14s per sample
- TRT-LLM accelerates the Qwen3 backbone (~50-60% of total inference time). The rest (audio codec, tokenization, embedding) stays in PyTorch.
- Larger speedups expected with batching, longer sequences, or bigger GPUs (A10G, A100).

### Reproduce

```bash
# Run TRT-LLM benchmark (build engine + test 10 samples)
modal run omnivoice_trtllm_modal.py

# Run PyTorch benchmark (edit main() to set use_trtllm=False)
# In omnivoice_trtllm_modal.py:
#   start_and_test_triton_server.remote(use_trtllm=False)
modal run omnivoice_trtllm_modal.py
```

### Voice samples

Test voices are in `voices/system/`. Each sample has a `.txt` and optionally a `.wav`:

```
voices/system/
├── vi_01-08       # Vietnamese voice clone (8 samples, wav + txt)
├── cross_01-03    # Cross-lingual (3 samples, wav + txt)
├── design_01-03   # Voice design (3 samples, txt only)
└── fine_01-03     # Fine-grained control (3 samples, txt only)
```

## Model Details

- **Model**: [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)
- **LLM backbone**: Qwen3 (28 layers, hidden=1024, 16 Q heads / 8 KV heads)
- **Audio codec**: HiggsAudioV2 (8 codebooks, vocab=1025)
- **GPU**: NVIDIA L4
- **Framework**: TensorRT-LLM 0.18.2 + PyTorch
- **Server**: NVIDIA Triton Server 25.04
- **Sample Rate**: 24kHz
