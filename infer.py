"""Text-to-audio inference for TinyAudio.

Checkpoints are downloaded from the Hugging Face Hub on first use.

Examples:
    # TinyAudio-MF, 1 step
    python infer.py --prompt "A dog is barking in a room"

    # TinyAudio (flow matching), 25 Euler steps
    python infer.py --variant tinyaudio --prompt "A dog is barking in a room"

    # batch generation from a JSON list of {"id": ..., "prompt_text": ...}
    python infer.py --json_path prompts.json --batch_size 8
"""
import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
from tqdm import tqdm

from tinyaudio.eval_utils import setup_eval_logging
from tinyaudio.pipeline import HF_REPO_ID, VARIANTS, TinyAudioPipeline

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()


def chunked(seq, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def output_stem(prompt: str) -> str:
    return prompt.replace(' ', '_').replace('/', '_').replace('.', '')[:200]


def load_prompts(args) -> list[tuple[str | None, str]]:
    if args.json_path is not None:
        with args.json_path.expanduser().open('r', encoding='utf-8') as f:
            items = json.load(f)
        prompts = [(it['id'], it['prompt_text']) for it in items if it.get('id') and it.get('prompt_text')]
        if not prompts:
            raise ValueError(f'No records with "id" and "prompt_text" in {args.json_path}')
        return prompts
    if not args.prompt:
        raise ValueError('Provide --prompt or --json_path')
    return [(None, args.prompt)]


def main():
    setup_eval_logging()

    parser = ArgumentParser()
    parser.add_argument('--variant', type=str, default='tinyaudio_mf', choices=list(VARIANTS),
                        help='model variant; selects config/<variant>.yaml')
    parser.add_argument('--config_name', type=str, default=None, help='override the config name under config/')
    parser.add_argument('--model_path', type=str, default=None,
                        help='network weights (.pth); defaults to the released checkpoint')
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--json_path', type=Path, default=None,
                        help='JSON list of {"id": ..., "prompt_text": ...} for batch generation')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--negative_prompt', type=str, default='')
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--cfg_strength', type=float, default=None, help='defaults to cfg_strength in the config')
    parser.add_argument('--num_steps', type=int, default=None, help='defaults to 1 (MF) or 25 (FM)')
    parser.add_argument('--flow_shift', type=float, default=None, help='timestep shift for flow-matching sampling')
    parser.add_argument('--sampling_method', type=str, default=None, help='euler / heun3 / dopri5 / adaptive')
    parser.add_argument('--cfg_interval_min', type=float, default=0.0, help='MeanFlow: lower bound of the CFG interval')
    parser.add_argument('--cfg_interval_max', type=float, default=1.0, help='MeanFlow: upper bound of the CFG interval')
    parser.add_argument('--text_encoder_attn_impl', type=str, default=None, help='sdpa / eager / flash_attention_2')
    parser.add_argument('--output', type=Path, default=Path('./output'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--full_precision', action='store_true', help='run in fp32 instead of bf16')
    parser.add_argument('--repo_id', type=str, default=HF_REPO_ID, help='Hugging Face repo to download checkpoints')
    parser.add_argument('--offline', action='store_true', help='use local or cached checkpoints only')
    args = parser.parse_args()

    pipe = TinyAudioPipeline.from_pretrained(args.variant,
                                             model_path=args.model_path,
                                             config_name=args.config_name,
                                             dtype=torch.float32 if args.full_precision else torch.bfloat16,
                                             duration=args.duration,
                                             repo_id=args.repo_id,
                                             local_files_only=args.offline,
                                             text_encoder_attn_impl=args.text_encoder_attn_impl)
    if pipe.device == 'cpu':
        log.warning('CUDA/MPS are not available, running on CPU')
    num_steps = args.num_steps if args.num_steps is not None else pipe.default_num_steps

    output_dir = args.output.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=pipe.device)
    generator.manual_seed(args.seed)

    prompts = load_prompts(args)
    for batch in tqdm(list(chunked(prompts, max(1, args.batch_size)))):
        ids = [b[0] for b in batch]
        texts = [b[1] for b in batch]
        log.info(f'Prompts: {texts}')
        audios = pipe(texts,
                      negative_prompt=args.negative_prompt,
                      num_steps=num_steps,
                      cfg_strength=args.cfg_strength,
                      generator=generator,
                      cfg_interval_min=args.cfg_interval_min,
                      cfg_interval_max=args.cfg_interval_max,
                      sampling_method=args.sampling_method,
                      flow_shift=args.flow_shift)

        for prompt_id, text, audio in zip(ids, texts, audios):
            if prompt_id is not None:
                save_path = output_dir / f'{prompt_id}.wav'
            else:
                save_path = output_dir / (f'{output_stem(text)}--numsteps{num_steps}--seed{args.seed}'
                                          f'--duration{args.duration}.wav')
            pipe.save(audio, save_path)
            log.info(f'Audio saved to {save_path}')

    if pipe.device == 'cuda':
        log.info('Memory usage: %.2f GB', torch.cuda.max_memory_allocated() / (2**30))


if __name__ == '__main__':
    main()
