<div align="center">

# TinyAudio
### Compact and Efficient Text-to-Audio Generation for Low-Resource Deployment

[![Project Page](https://img.shields.io/badge/Website-Project_Page-orange?logo=googlechrome&logoColor=white)](https://tinyaudio-project.github.io/)
![Code](https://img.shields.io/badge/Code-Coming_Soon-lightgrey)
![Models](https://img.shields.io/badge/Models-Coming_Soon-lightgrey)

</div>

## Overview

TinyAudio is a compact flow-matching model for text-to-audio generation on resource-constrained devices. It combines audio-aligned text conditioning, a single-stream generative Transformer, and a lightweight waveform decoder to generate 44.1 kHz audio with **87M inference-time parameters**.

TinyAudio-MF extends the same architecture with MeanFlow training for one-step generation. It achieves a mean steady-state real-time factor (RTF) of **0.715** when generating 10-second audio under a four-core CPU quota.

> **Release status:** This repository currently hosts the project introduction. Source code, pretrained checkpoints, and runnable instructions are not yet available.

## Highlights- **Efficient architecture:** Single-stream text–audio modeling with layer-shared conditional modulation.
- **Quality-aware training:** Supervised fine-tuning prioritizes perceptually strong and semantically consistent audio–text pairs.- **One-step generation:** TinyAudio-MF enables real-time CPU generation.

## Model Variants| Model | Sampling | Audio sample rate | Release || --- | --- | --- | --- |
| TinyAudio | 25-step Euler with CFG | 44.1 kHz | Coming soon |
| TinyAudio-MF | One-step MeanFlow | 44.1 kHz | Coming soon |

## Results

AudioCaps generation results reported in the manuscript:

| Model | FAD ↓ | FD ↓ | KL ↓ | IS ↑ |
| --- | --- | --- | --- | --- |
| TinyAudio | 2.65 | 18.24 | 1.32 | 10.44 |
| TinyAudio-MF | 2.83 | 23.65 | 1.49 | 8.47 |

TinyAudio uses 0.48 GB peak GPU memory in FP32 at batch size 1. The CPU result above was measured on a dual-socket Intel Xeon Gold 6530 node with a CPU quota equivalent to four cores, using FP32 and batch size 1. Timing includes text encoding and waveform decoding, but excludes model loading and audio saving.

## Environmental Setup

Coming soon. Tested dependencies and installation instructions will accompany the code release.

## Quick Start

Coming soon. Inference examples and checkpoint download links will be added when the models are released.

Project website: [tinyaudio-project.github.io](https://tinyaudio-project.github.io/).

## Training and Evaluation

Training configurations, data-preparation instructions, and evaluation commands are being prepared for release.

## Citation

Paper link and BibTeX entry will be added when the manuscript is publicly available.
