#!/usr/bin/env bash
# Generate audio with TinyAudio (flow matching, 25 Euler steps).
set -e

python infer.py \
    --variant tinyaudio \
    --model_path ./weights/tinyaudio.pth \
    --prompt "A dog is barking while a car passes by" \
    --num_steps 25 \
    --cfg_strength 9 \
    --duration 10 \
    --seed 42 \
    --output ./output/tinyaudio
