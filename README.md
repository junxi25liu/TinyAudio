# TinyAudio

> Small model. Fast sound. Big ambition.

TinyAudio CPU is a minimal ONNX Runtime inference package for text-to-audio generation on CPU.

The runtime is intentionally small: `tinyaudio.py` contains the complete Python API and CLI.

- `standard/`: FP32 ONNX package for CPU inference.

Before running inference, place the exported ONNX assets and tokenizer files in `standard/`. The directory should contain a `pipeline.json`, the ONNX files referenced by it, and a local `tokenizer/` directory.

## Install

```bash
pip install -r requirements.txt
```

The default provider mode is `auto`: it uses CUDA when available, otherwise CPU.

For GPU inference, install ONNX Runtime GPU instead:

```bash
pip uninstall -y onnxruntime onnxruntime-openvino
pip install numpy transformers onnxruntime-gpu
```

Provider selection is automatic by default:

```text
CUDAExecutionProvider -> CPUExecutionProvider
```

OpenVINO can still be selected manually with `--ort_provider OpenVINOExecutionProvider --decoder_ort_provider OpenVINOExecutionProvider` when your environment supports it reliably.

## Quick Start

```bash
python tinyaudio.py "A dog is barking in a room"
```

The generated WAV path is printed at the end. By default, files are written to `out/`.

Choose an output path:

```bash
python tinyaudio.py "Rain falling on a window" --output out/rain.wav
```

## Python API

```python
from tinyaudio import TinyAudio

model = TinyAudio()
wav_path = model.generate("A dog is barking in a room", "out/dog.wav")
print(wav_path)
```

Single-call helper:

```python
from tinyaudio import tinyaudio

tinyaudio("Birds chirping in a forest", "out/birds.wav")
```

Explicit model directory:

```python
from tinyaudio import TinyAudio

model = TinyAudio(model_dir="standard")
model.generate("A dog is barking in a room", "out/dog_fp32.wav")
```

## CLI Options

```bash
python tinyaudio.py "A dog is barking in a room" \
  --mode standard \
  --num_steps 4 \
  --cfg_strength 7.0 \
  --seed 123
```

Common options:

- `--mode standard`: bundled FP32 mode.
- `--model_dir PATH`: use a custom exported model package.
- `--num_steps 4`: MeanFlow sampling steps.
- `--cfg_strength 7.0`: classifier-free guidance strength.
- `--output out/example.wav`: output WAV path.
- `--ort_provider auto`: use GPU when available, otherwise CPU.
- `--decoder_ort_provider auto`: provider for waveform decoding.
- `--verbose`: print model loading details and stage timing.

CFG interval is fixed to `[0.0, 1.0]` inside `tinyaudio.py`.

## Benchmark

```bash
python tinyaudio.py "A dog is barking in a room" \
  --mode standard \
  --warmup_runs 1 \
  --repeat_runs 3 \
  --verbose
```

## Environment Variables

```bash
CPU_NUM_THREADS=16
CPU_NUM_INTEROP_THREADS=1
ORT_PROVIDER=auto
ORT_DECODER_PROVIDER=auto
TINY_AUDIO_FRONT_PRECISION=fp32
```

For most users, the defaults are sufficient.

## Package Layout

```text
TinyAudio-cpu/
  tinyaudio.py
  requirements.txt
  README.md
  standard/
    README.md
    pipeline.json
    tokenizer/
    *.onnx
```
