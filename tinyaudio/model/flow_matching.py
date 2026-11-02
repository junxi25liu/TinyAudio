import logging
from typing import Callable, Optional

import torch
from torchdiffeq import odeint

log = logging.getLogger()

_ADAPTIVE_SOLVERS = {'adaptive', 'dopri5', 'dopri8', 'bosh3', 'adaptive_heun'}


## partially from https://github.com/gle-bellier/flow-matching
class FlowMatching:

    def __init__(self, min_sigma: float = 0.0,
                 inference_mode='euler',
                 num_steps: int = 25,
                 reverse_flow: bool = True,
                 shift: float = 1.0,
                 rtol: float = 1e-3,
                 atol: float = 1e-3):
        # inference_mode: 'euler', 'heun3', 'adaptive', or an explicit torchdiffeq solver such as 'dopri5'
        # num_steps: number of steps in the fixed-step inference modes
        super().__init__()
        self.min_sigma = min_sigma
        self.inference_mode = inference_mode
        self.num_steps = num_steps
        self.reverse_flow = reverse_flow
        self.shift = float(shift)
        self.rtol = float(rtol)
        self.atol = float(atol)

        assert self.inference_mode in ['euler', 'heun3', *_ADAPTIVE_SOLVERS]
        if self.inference_mode in _ADAPTIVE_SOLVERS and num_steps > 0:
            log.info('The number of steps is ignored in adaptive inference mode ')

    def _apply_shift(self, steps: torch.Tensor) -> torch.Tensor:
        if self.shift == 1.0:
            return steps
        if torch.any(steps < 0) or torch.any(steps > 1):
            log.warning(
                'FlowMatching shift expects steps in [0, 1], got range [%.6f, %.6f]; skipping shift.',
                float(steps.min().item()),
                float(steps.max().item()),
            )
            return steps
        shift = steps.new_tensor(self.shift)
        return (shift * steps) / (1.0 + (shift - 1.0) * steps)

    def get_conditional_flow(self, x0: torch.Tensor, x1: torch.Tensor,
                             t: torch.Tensor) -> torch.Tensor:
        # which is psi_t(x), eq 22 in flow matching for generative models
        t = t[:, None, None].expand_as(x0)
        if self.reverse_flow:
            return (1 - t) * x1 + t * x0  # xt = (1-t)*x1 + t*x0 -> vt = x0 - x1
        else:
            return (1 - t) * x0 + t * x1  # xt = (1-t)*x0 + t*x1 -> vt = x1 - x0

    def loss(self, predicted_v: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        # return the mean error without reducing the batch dimension
        reduce_dim = list(range(1, len(predicted_v.shape)))
        if self.reverse_flow:
            target_v = x0 - x1
        else:
            target_v = x1 - x0
        return (predicted_v - target_v).pow(2).mean(dim=reduce_dim)

    def get_x0_xt_c(
        self,
        x1: torch.Tensor,
        t: torch.Tensor,
        Cs: list[torch.Tensor],
        generator: Optional[torch.Generator] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x0 = torch.empty_like(x1).normal_(generator=generator)

        xt = self.get_conditional_flow(x0, x1, t)
        return x0, x1, xt, Cs

    def to_prior(self, fn: Callable, x1: torch.Tensor) -> torch.Tensor:
        if self.reverse_flow:
            return self.run_t0_to_t1(fn, x1, 0, 1)
        else:
            return self.run_t0_to_t1(fn, x1, 1, 0)

    def to_data(self, fn: Callable, x0: torch.Tensor) -> torch.Tensor:
        if self.reverse_flow:
            return self.run_t0_to_t1(fn, x0, 1, 0)
        else:
            return self.run_t0_to_t1(fn, x0, 0, 1)

    def run_t0_to_t1(self, fn: Callable, x0: torch.Tensor, t0: float, t1: float) -> torch.Tensor:
        # fn: a function that takes (t, x) and returns the direction x0->x1

        if self.inference_mode in _ADAPTIVE_SOLVERS:
            method = 'dopri5' if self.inference_mode == 'adaptive' else self.inference_mode
            return odeint(
                fn,
                x0,
                torch.tensor([t0, t1], device=x0.device, dtype=x0.dtype),
                method=method,
                rtol=self.rtol,
                atol=self.atol,
            )[-1]
        elif self.inference_mode == 'euler':
            x = x0
            steps = torch.linspace(t0, t1 - self.min_sigma, self.num_steps + 1, device=x0.device, dtype=x0.dtype)
            steps = self._apply_shift(steps)
            for ti, t in enumerate(steps[:-1]):
                flow = fn(t, x)
                next_t = steps[ti + 1]
                dt = next_t - t
                x = x + dt * flow
            return x
        elif self.inference_mode == 'heun3':
            x = x0
            steps = torch.linspace(t0, t1 - self.min_sigma, self.num_steps + 1, device=x0.device, dtype=x0.dtype)
            steps = self._apply_shift(steps)
            for ti, t in enumerate(steps[:-1]):
                next_t = steps[ti + 1]
                dt = next_t - t

                k1 = fn(t, x)

                t2 = t + dt / 3
                x2 = x + (dt / 3) * k1
                k2 = fn(t2, x2)

                t3 = t + (2 * dt / 3)
                x3 = x + (2 * dt / 3) * k2
                k3 = fn(t3, x3)

                # Third-order Heun method.
                x = x + dt * (k1 + 3 * k3) / 4
            return x
