#!/usr/bin/env bash
# Evaluate TinyAudio-MF on the AudioCaps test split.
# Metrics need av-benchmark and AudioCaps_test.gt_cache (see config/data/tinyaudio_mix.yaml);
# without them the script only writes the generated audio.
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')

OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=$NUM_GPUS \
    eval.py \
    --config-name tinyaudio_mf \
    exp_id=eval_tinyaudio_mf \
    weights=./weights/tinyaudio_mf.pth \
    eval_batch_size=32
