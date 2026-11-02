#!/usr/bin/env bash
# Train TinyAudio-MF (MeanFlow, 1-step sampling).
# Prepare the JSONL files listed in config/data/tinyaudio_mix.yaml first.
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')

btz=128
num_iterations=200_000
exp_id=tinyaudio_mf_bsz${btz}_ngpu${NUM_GPUS}

OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=$NUM_GPUS \
    train.py \
    --config-name tinyaudio_mf \
    exp_id=$exp_id \
    batch_size=$btz \
    num_iterations=$num_iterations \
    num_workers=10
