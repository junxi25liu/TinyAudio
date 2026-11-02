#!/usr/bin/env bash
# Generate audio with TinyAudio-MF (1 step).
set -e

python infer.py \
    --variant tinyaudio_mf \
    --model_path ./weights/tinyaudio_mf.pth \
    --prompt "A dog is barking while a car passes by" \
    --num_steps 1 \
    --cfg_strength 7 \
    --duration 10 \
    --seed 42 \
    --output ./output/tinyaudio_mf
