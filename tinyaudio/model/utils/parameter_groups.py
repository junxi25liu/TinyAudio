import logging

import torch.optim as optim

log = logging.getLogger()


def get_parameter_groups(model, cfg):
    params = []
    memo = set()
    for _, param in model.named_parameters():
        if not param.requires_grad or param in memo:
            continue
        memo.add(param)
        params.append(param)
    return [{'params': params, 'lr': cfg.learning_rate, 'weight_decay': cfg.weight_decay}]


def build_optimizer(model, cfg, use_amp: bool = False):
    return optim.AdamW(get_parameter_groups(model, cfg),
                       lr=cfg.learning_rate,
                       weight_decay=cfg.weight_decay,
                       betas=[0.9, 0.95],
                       eps=1e-6 if use_amp else 1e-8,
                       fused=True)
