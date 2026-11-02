import logging
import math
from functools import partial
from typing import Callable, Optional

import numpy as np
import torch
from einops import rearrange

from tinyaudio.model.transformer_layers import disable_flash_attention

log = logging.getLogger()


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.mean(error ** 2, dim=(1, 2), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return stopgrad(w) * loss


def cosine_annealing(start, end, step, total_steps):
    cos_inner = math.pi * step / total_steps
    return end + 0.5 * (start - end) * (1 + math.cos(cos_inner))


## partially from https://github.com/haidog-yaqub/MeanFlow
class MeanFlow():
    def __init__(
        self,
        steps=1,
        flow_ratio=0.75,
        time_dist=['lognorm', -0.4, 1.0],
        jvp_api='autograd',
        null_condition_probability=0.0,
        flexible_cfg=False,
        cfg_scale_max=8.0,
        cfg_scale_beta=1.0,
        shift_enabled: bool = False,
        shift: float = 1.0,
    ):
        super().__init__()
        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.steps = steps
        self.jvp_api = jvp_api
        self.null_condition_probability = null_condition_probability
        self.flexible_cfg = flexible_cfg
        self.cfg_scale_max = cfg_scale_max
        self.cfg_scale_beta = cfg_scale_beta
        self.shift_enabled = bool(shift_enabled)
        self.shift = float(shift)
        assert jvp_api in ['funtorch', 'autograd'], "jvp_api must be 'funtorch' or 'autograd'"
        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif jvp_api == 'autograd':
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True
        log.info(f'MeanFlow initialized with {steps} steps, shift_enabled={self.shift_enabled}, shift={self.shift}')

    def _apply_shift(self, steps: torch.Tensor) -> torch.Tensor:
        if not self.shift_enabled or self.shift == 1.0:
            return steps
        if torch.any(steps < 0) or torch.any(steps > 1):
            log.warning(
                'MeanFlow shift expects steps in [0, 1], got range [%.6f, %.6f]; skipping shift.',
                float(steps.min().item()),
                float(steps.max().item()),
            )
            return steps
        shift = steps.new_tensor(self.shift)
        return (shift * steps) / (1.0 + (shift - 1.0) * steps)

    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))

        samples = torch.tensor(samples, device=device)
        samples = self._apply_shift(samples)

        t = torch.maximum(samples[:, 0], samples[:, 1])
        r = torch.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        indices = torch.randperm(batch_size, device=device)[:num_selected]
        r[indices] = t[indices]
        return t, r

    def sample_cfg_scale(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.flexible_cfg:
            return torch.ones(batch_size, device=device, dtype=dtype)

        if self.cfg_scale_max <= 1.0:
            return torch.ones(batch_size, device=device, dtype=dtype)

        u = torch.rand(batch_size, device=device, dtype=dtype)
        scale_min = torch.tensor(1.0, device=device, dtype=dtype)
        scale_max = torch.tensor(self.cfg_scale_max, device=device, dtype=dtype)

        if self.cfg_scale_beta == 1.0:
            log_span = torch.log(scale_max / scale_min)
            return scale_min * torch.exp(u * log_span)

        beta = torch.tensor(self.cfg_scale_beta, device=device, dtype=dtype)
        min_pow = scale_min.pow(1.0 - beta)
        max_pow = scale_max.pow(1.0 - beta)
        mixed = min_pow + u * (max_pow - min_pow)
        return mixed.pow(1.0 / (1.0 - beta))

    def sample_cfg_interval(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.flexible_cfg:
            return (
                torch.zeros(batch_size, device=device, dtype=dtype),
                torch.ones(batch_size, device=device, dtype=dtype),
            )

        t_min = torch.empty(batch_size, device=device, dtype=dtype).uniform_(0.0, 0.5)
        t_max = torch.empty(batch_size, device=device, dtype=dtype).uniform_(0.5, 1.0)
        return t_min, t_max

    def to_prior(self, fn: Callable, x1: torch.Tensor) -> torch.Tensor:
        return self.run_t0_to_t1(fn, x1)

    @torch.no_grad()
    def to_data(self, fn: Callable, x0: torch.Tensor) -> torch.Tensor:
        return self.run_t0_to_t1(fn, x0)

    def run_t0_to_t1(self, fn: Callable, x0: torch.Tensor) -> torch.Tensor:
        steps = torch.linspace(1, 0, self.steps + 1).to(device=x0.device, dtype=x0.dtype)
        steps = self._apply_shift(steps)
        for ti, t in enumerate(steps[:-1]):
            t = t.expand(x0.shape[0])
            next_t = steps[ti + 1].expand(x0.shape[0])
            u_flow = fn(t=t, r=next_t, x=x0)
            dt = (t - next_t).mean()
            x0 = x0 - dt * u_flow
        return x0

    def loss(self,
            fn: Callable,
            x0: torch.Tensor,
            text_f: torch.Tensor,
            text_f_c: torch.Tensor,
            empty_string_feat: torch.Tensor,
            empty_string_feat_c: torch.Tensor,
            text_padding_mask: Optional[torch.Tensor] = None,
            empty_string_padding_mask: Optional[torch.Tensor] = None):

        batch_size = x0.shape[0]
        device = x0.device
        e = torch.randn_like(x0)
        t, r = self.sample_t_r(batch_size, device)
        t_ = rearrange(t, "b -> b 1 1 ")
        r_ = rearrange(r, "b -> b 1 1 ")
        z_t = (1 - t_) * x0 + t_ * e  # r < t
        v_t = e - x0
        omega = self.sample_cfg_scale(batch_size, device, x0.dtype)
        t_min, t_max = self.sample_cfg_interval(batch_size, device, x0.dtype)
        w = torch.where((t >= t_min) & (t <= t_max), omega, torch.ones_like(omega))
        text_f_u = empty_string_feat.expand(batch_size, -1, -1)
        text_f_c_u = empty_string_feat_c.expand(batch_size, -1)
        text_padding_mask_u = None
        if empty_string_padding_mask is not None:
            text_padding_mask_u = empty_string_padding_mask.to(device=device, dtype=torch.bool).view(1, -1).expand(batch_size, -1)

        v_u = fn(
            latent=z_t,
            text_f=text_f_u,
            text_f_c=text_f_c_u,
            r=t,
            t=t,
            omega=w,
            t_min=t_min,
            t_max=t_max,
            text_padding_mask=text_padding_mask_u,
        ).detach().requires_grad_(False)
        v_c = fn(
            latent=z_t,
            text_f=text_f,
            text_f_c=text_f_c,
            r=t,
            t=t,
            omega=w,
            t_min=t_min,
            t_max=t_max,
            text_padding_mask=text_padding_mask,
        ).detach().requires_grad_(False)
        v_g = v_t + rearrange(1.0 - 1.0 / w.clamp_min(1.0), "b -> b 1 1") * (v_c - v_u)

        if self.null_condition_probability > 0:
            drop_mask = torch.rand(batch_size, device=device) < self.null_condition_probability
            if drop_mask.any():
                text_f = text_f.clone()
                text_f_c = text_f_c.clone()
                text_f[drop_mask] = empty_string_feat
                text_f_c[drop_mask] = empty_string_feat_c
                if text_padding_mask is not None and empty_string_padding_mask is not None:
                    text_padding_mask = text_padding_mask.clone()
                    text_padding_mask[drop_mask] = empty_string_padding_mask
                v_g = torch.where(drop_mask.view(batch_size, 1, 1), v_t, v_g)

        device = z_t.device
        model_partial = partial(
            fn,
            text_f=text_f,
            text_f_c=text_f_c,
            omega=w,
            t_min=t_min,
            t_max=t_max,
            text_padding_mask=text_padding_mask,
        )
        jvp_args = (
            lambda z_f, r_f, t_f: model_partial(latent=z_f, r=r_f, t=t_f),
            (z_t, r, t),
            (v_c, torch.zeros_like(r), torch.ones_like(t)),
        )
        with disable_flash_attention():
            if self.create_graph:
                u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
            else:
                u, dudt = self.jvp_fn(*jvp_args)
        V = u + (t_ - r_) * stopgrad(dudt)
        error = V - stopgrad(v_g)
        loss = adaptive_l2_loss(error)
        return loss, r, t

