"""Training / evaluation runners for TinyAudio.

``RunnerFlowMatching`` trains the flow-matching model (``tinyaudio``) and
``RunnerMeanFlow`` trains the MeanFlow model (``tinyaudio_mf``). Both share the
feature extractors, optimisation, logging and checkpointing code in ``Runner``.
"""
import os
from pathlib import Path
from typing import Optional, Union

import torch
import torch.distributed
import torch.optim as optim
from nitrous_ema import PostHocEMA
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from tinyaudio.eval_utils import generate_fm, generate_mf, load_text_condition
from tinyaudio.model.flow_matching import FlowMatching
from tinyaudio.model.mean_flow import MeanFlow
from tinyaudio.model.networks import build_model_from_cfg
from tinyaudio.model.sequence_config import resolve_sequence_config_from_cfg
from tinyaudio.model.utils.features_utils import FeaturesUtils
from tinyaudio.model.utils.parameter_groups import build_optimizer
from tinyaudio.utils.dist_utils import info_if_rank_zero, local_rank
from tinyaudio.utils.log_integrator import Integrator
from tinyaudio.utils.logger import TensorboardLogger
from tinyaudio.utils.time_estimator import PartialTimeEstimator, TimeEstimator

TEXT_CONDITION_PARAMS = ('learned_text_pad_token', 'text_token_pool.weight', 'text_token_pool.bias')


def log_normal_sample(x: torch.Tensor, generator: Optional[torch.Generator] = None, m: float = 0.0,
                      s: float = 1.0) -> torch.Tensor:
    """Logit-normal time sampling."""
    return torch.sigmoid(torch.randn(x.shape[0], device=x.device, generator=generator) * s + m)


class Runner:

    def __init__(self,
                 cfg: DictConfig,
                 log: TensorboardLogger,
                 run_path: Union[str, Path],
                 for_training: bool = True,
                 latent_mean: Optional[torch.Tensor] = None,
                 latent_std: Optional[torch.Tensor] = None):
        self.cfg = cfg
        self.exp_id = cfg.exp_id
        self.use_amp = cfg.amp
        self.enable_grad_scaler = cfg.get('enable_grad_scaler', True)
        self.for_training = for_training
        self.log = log
        self.run_path = Path(run_path)

        self.seq_cfg = resolve_sequence_config_from_cfg(cfg)
        self.sample_rate = self.seq_cfg.sampling_rate
        self.duration_sec = self.seq_cfg.duration
        info_if_rank_zero(log, f'latent_seq_len={self.seq_cfg.latent_seq_len}, hop={self.seq_cfg.hop_length}')

        # frozen text encoder (TA-CLAP) and audio VAE (TA-VAE)
        self.use_text_mask = bool(cfg.get('use_learned_text_pad_token', True))
        self.features = FeaturesUtils(vae_ckpt=cfg.vae_ckpt,
                                      vae_model_tag=cfg.vae_model_tag,
                                      vae_use_ema=cfg.vae_use_ema,
                                      enable_conditions=True,
                                      text_encoder_path=cfg.text_encoder_path,
                                      text_encoder_weights_path=cfg.text_encoder_ckpt,
                                      text_encoder_attn_implementation=cfg.get('text_encoder_attn_implementation',
                                                                               None),
                                      need_vae_encoder=for_training).cuda().eval()
        if cfg.compile:
            self.features.compile()

        with torch.no_grad():
            empty_feat, empty_feat_c, empty_mask = self.features.encode_text([''], return_padding_mask=True)

        # network
        network = build_model_from_cfg(cfg,
                                       latent_mean=latent_mean,
                                       latent_std=latent_std,
                                       empty_string_feat=empty_feat[0],
                                       empty_string_feat_c=empty_feat_c[0],
                                       empty_string_padding_mask=empty_mask[0] if self.use_text_mask else None).cuda()
        network.update_seq_lengths(self.seq_cfg.latent_seq_len)
        if self.use_text_mask:
            # padding token and token pooling were aligned together with TA-CLAP; keep them frozen
            load_text_condition(network, cfg.text_encoder_ckpt)
            named_params = dict(network.named_parameters())
            for name in TEXT_CONDITION_PARAMS:
                if name in named_params:
                    named_params[name].requires_grad = False
        self.network = DDP(network, device_ids=[local_rank], broadcast_buffers=False)

        # EMA
        if for_training and cfg.ema.enable and local_rank == 0:
            self.ema = PostHocEMA(self.network.module,
                                  sigma_rels=cfg.ema.sigma_rels,
                                  update_every=cfg.ema.update_every,
                                  checkpoint_every_num_steps=cfg.ema.checkpoint_every,
                                  checkpoint_folder=cfg.ema.checkpoint_folder,
                                  step_size_correction=True).cuda()
            self.ema_start = cfg.ema.start
        else:
            self.ema = None

        self.rng = torch.Generator(device='cuda')
        self.rng.manual_seed(cfg.seed + local_rank)

        self.null_condition_probability = cfg.null_condition_probability
        self.cfg_strength = cfg.cfg_strength
        self.setup_sampler(cfg)

        total = sum(p.nelement() for p in self.network.parameters())
        trainable = sum(p.nelement() for p in self.network.parameters() if p.requires_grad)
        info_if_rank_zero(log, f'Network parameters: total={total / 1e6:.2f}M, trainable={trainable / 1e6:.2f}M')
        info_if_rank_zero(log, 'torch version: ' + torch.__version__)

        self.train_integrator = Integrator(self.log, distributed=True)
        self.val_integrator = Integrator(self.log, distributed=True)

        if for_training:
            self.enter_train()
            self.setup_optimization(cfg)
        else:
            self.enter_val()

    # ------------------------------------------------------------------ setup
    def setup_sampler(self, cfg: DictConfig) -> None:
        raise NotImplementedError

    def setup_optimization(self, cfg: DictConfig) -> None:
        self.optimizer = build_optimizer(self.network, cfg, use_amp=self.use_amp)
        if self.enable_grad_scaler:
            self.scaler = torch.amp.GradScaler(init_scale=2048)
        self.clip_grad_norm = cfg.clip_grad_norm

        warmup_steps = cfg.linear_warmup_steps
        warmup = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: (step + 1) / (warmup_steps + 1))
        if cfg.lr_schedule == 'constant':
            main = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda _: 1)
        elif cfg.lr_schedule == 'poly':
            total = cfg.num_iterations
            main = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda x: (1 - (x / total))**0.9)
        elif cfg.lr_schedule == 'step':
            main = optim.lr_scheduler.MultiStepLR(self.optimizer, list(cfg.lr_schedule_steps), cfg.lr_schedule_gamma)
        else:
            raise NotImplementedError(f'Unknown lr_schedule: {cfg.lr_schedule}')
        self.scheduler = optim.lr_scheduler.SequentialLR(self.optimizer, [warmup, main], [warmup_steps])

        self.log_text_interval = cfg.log_text_interval
        self.log_extra_interval = cfg.log_extra_interval
        self.save_weights_interval = cfg.save_weights_interval
        self.save_checkpoint_interval = cfg.save_checkpoint_interval
        self.save_copy_iterations = list(cfg.save_copy_iterations)
        self.num_iterations = cfg.num_iterations

        self.log.batch_timer = TimeEstimator(self.num_iterations, self.log_text_interval)
        self.log.data_timer = PartialTimeEstimator(self.num_iterations, 1, ema_alpha=0.9)

    # ------------------------------------------------------------ subclasses
    def compute_loss(self, text_f, text_f_c, text_padding_mask, x1, loss_mask=None):
        """Return (per-sample loss, mean loss, sampled time) for normalised latents x1."""
        raise NotImplementedError

    def generate(self, captions: list[str]) -> torch.Tensor:
        """Generate waveforms for a list of captions."""
        raise NotImplementedError

    # ---------------------------------------------------------------- helpers
    def encode_text(self, captions: list[str]):
        if self.use_text_mask:
            return self.features.encode_text(captions, return_padding_mask=True)
        text_f, text_f_c = self.features.encode_text(captions)
        return text_f, text_f_c, None

    @torch.no_grad()
    def encode_batch(self, data):
        waveforms = data['waveforms'].cuda(non_blocking=True)
        text_f, text_f_c, text_padding_mask = self.encode_text(data['caption'])
        dist = self.features.encode_audio(waveforms)
        a_mean = dist.mean.transpose(1, 2)
        a_std = dist.std.transpose(1, 2)
        x1 = a_mean + a_std * torch.empty_like(a_mean).normal_(generator=self.rng)
        return text_f, text_f_c, text_padding_mask, x1

    def make_loss_mask(self, data, latent_seq_len: int, device) -> Optional[torch.Tensor]:
        lengths = data.get('waveform_lengths', None)
        if lengths is None:
            return None
        lengths = torch.as_tensor(lengths, device=device, dtype=torch.float32)
        valid = (lengths / self.seq_cfg.hop_length).ceil().long().clamp(max=latent_seq_len)
        indices = torch.arange(latent_seq_len, device=device).unsqueeze(0)
        return (indices < valid.unsqueeze(1)).float()

    # ---------------------------------------------------------------- training
    def train_pass(self, data, it: int = 0):
        if not self.for_training:
            raise ValueError('train_pass() should not be called when not training.')

        self.enter_train()
        self.log.data_timer.end()
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
            text_f, text_f_c, text_padding_mask, x1 = self.encode_batch(data)
            loss_mask = self.make_loss_mask(data, x1.shape[1], x1.device)
            x1 = self.network.module.normalize(x1)
            loss, mean_loss, t = self.compute_loss(text_f, text_f_c, text_padding_mask, x1, loss_mask)
            self.train_integrator.add_dict({'loss': mean_loss})

        if it % self.log_text_interval == 0 and it != 0:
            self.train_integrator.add_scalar('lr', self.scheduler.get_last_lr()[0])
            self.train_integrator.add_binned_tensor('binned_loss', loss.detach(), t)
            self.train_integrator.finalize('train', it)
            self.train_integrator.reset_except_hooks()

        # backward
        self.optimizer.zero_grad(set_to_none=True)
        if self.enable_grad_scaler:
            self.scaler.scale(mean_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            mean_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad_norm)
            self.optimizer.step()

        if self.ema is not None and it >= self.ema_start:
            self.ema.update()
        self.scheduler.step()
        self.integrator.add_scalar('grad_norm', grad_norm)

        if it % self.log_extra_interval == 0:
            self.log_samples(data, x1, it)

        save_copy = it in self.save_copy_iterations
        if (it % self.save_weights_interval == 0 and it != 0) or save_copy:
            self.save_weights(it)
        if it % self.save_checkpoint_interval == 0 and it != 0:
            self.save_checkpoint(it, save_copy=save_copy)

        self.log.data_timer.start()

    @torch.inference_mode()
    def log_samples(self, data, x1: torch.Tensor, it: int) -> None:
        """Write one ground-truth reconstruction and one generated sample to the run directory."""
        self.enter_val()
        try:
            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
                gt = self.network.module.unnormalize(x1[0:1].clone())
                audio = self.features.vocode(self.features.decode(gt)).float().cpu()[0]
                self.log.log_audio('train', f'audio-gt-r{local_rank}', audio.view(1, -1), it,
                                   sample_rate=self.sample_rate)
                audio = self.generate([data['caption'][0]]).float().cpu()[0]
                self.log.log_audio('train', f'audio-r{local_rank}', audio.view(1, -1), it,
                                   sample_rate=self.sample_rate)
        except Exception as e:
            self.log.warning(f'Error when logging samples: {e}')
            if self.cfg.debug:
                raise
        self.enter_train()

    # --------------------------------------------------------------- inference
    @torch.inference_mode()
    def inference_pass(self, data, it: int, data_cfg: DictConfig, *, save_eval: bool = True) -> Path:
        """Generate audio for a batch of captions and write it to the run directory."""
        self.enter_val()
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
            audio = self.generate(list(data['caption'])).float().cpu()
        audio_dir = None
        for i in range(audio.shape[0]):
            waveform = audio[i].view(1, -1)
            if data_cfg.get('output_subdir', None) is not None:
                prefix = f'{it:09d}' if save_eval else 'val-cache'
                audio_dir = self.log.log_audio(prefix, str(data['id'][i]), waveform, it=None,
                                               sample_rate=self.sample_rate, subdir=Path(data_cfg.output_subdir))
            else:
                audio_dir = self.log.log_audio(f'{data_cfg.tag}-sampled', str(data['id'][i]), waveform, it=None,
                                               sample_rate=self.sample_rate)
        return Path(audio_dir)

    @torch.inference_mode()
    def eval(self, audio_dir: Path, it: int, data_cfg: DictConfig) -> Optional[dict[str, float]]:
        """Compute metrics with av-benchmark (https://github.com/hkchengrex/av-benchmark), if installed."""
        if local_rank != 0:
            return None
        gt_cache = data_cfg.get('gt_cache', None)
        if gt_cache is None:
            self.log.warning('data_cfg.gt_cache is not set; skipping metric computation.')
            return None
        try:
            from av_bench.evaluate import evaluate
            from av_bench.extract import extract
        except ImportError:
            self.log.warning('av-benchmark is not installed; skipping metric computation.')
            return None

        prev_deterministic = torch.backends.cudnn.deterministic
        prev_benchmark = torch.backends.cudnn.benchmark
        audio_dir = Path(audio_dir).resolve()
        try:
            with torch.amp.autocast('cuda', enabled=False):
                extract(audio_path=audio_dir,
                        output_path=audio_dir / 'cache',
                        device='cuda',
                        batch_size=16,
                        num_workers=4,
                        skip_video_related=True,
                        audio_length=self.duration_sec)
                metrics = evaluate(gt_audio_cache=Path(gt_cache).resolve(),
                                   pred_audio_cache=audio_dir / 'cache',
                                   skip_video_related=True)
            for k, v in metrics.items():
                self.log.log_scalar(f'{data_cfg.tag}/{k}', v, it)
                self.log.info(f'{data_cfg.tag}/{k:<10}: {v:.10f}')
            return metrics
        finally:
            # av-benchmark changes global cuDNN flags at import time
            torch.backends.cudnn.deterministic = prev_deterministic
            torch.backends.cudnn.benchmark = prev_benchmark

    # ----------------------------------------------------------- checkpoints
    def save_weights(self, it, save_copy=True):
        if local_rank != 0:
            return
        os.makedirs(self.run_path, exist_ok=True)
        if save_copy:
            model_path = self.run_path / f'{self.exp_id}_{it}.pth'
            torch.save(self.network.module.state_dict(), model_path)
            self.log.info(f'Network weights saved to {model_path}.')

        model_path = self.run_path / f'{self.exp_id}_last.pth'
        if model_path.exists():
            model_path.replace(model_path.with_name(model_path.name.replace('last', 'prev')))
        torch.save(self.network.module.state_dict(), model_path)
        self.log.info(f'Network weights saved to {model_path}.')

    def save_checkpoint(self, it, save_copy=False):
        if local_rank != 0:
            return
        checkpoint = {
            'it': it,
            'weights': self.network.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'ema': self.ema.state_dict() if self.ema is not None else None,
        }
        os.makedirs(self.run_path, exist_ok=True)
        if save_copy:
            model_path = self.run_path / f'{self.exp_id}_ckpt_{it}.pth'
            torch.save(checkpoint, model_path)
            self.log.info(f'Checkpoint saved to {model_path}.')

        model_path = self.run_path / f'{self.exp_id}_ckpt_last.pth'
        if model_path.exists():
            model_path.replace(model_path.with_name(model_path.name.replace('last', 'prev')))
        torch.save(checkpoint, model_path)
        self.log.info(f'Checkpoint saved to {model_path}.')

    def _existing(self, name: str) -> Optional[Path]:
        path = self.run_path / name
        if not path.exists():
            info_if_rank_zero(self.log, f'No file found at {path}.')
            return None
        return path

    def get_latest_checkpoint_path(self):
        return self._existing(f'{self.exp_id}_ckpt_last.pth')

    def get_latest_weight_path(self):
        return self._existing(f'{self.exp_id}_last.pth')

    def get_final_ema_weight_path(self):
        return self._existing(f'{self.exp_id}_ema_final.pth')

    def load_checkpoint(self, path):
        map_location = f'cuda:{local_rank}'
        checkpoint = torch.load(path, map_location={'cuda:0': map_location}, weights_only=True)
        it = checkpoint['it']
        if self.ema is not None and checkpoint.get('ema') is not None:
            self.ema.load_state_dict(checkpoint['ema'])
            self.log.info(f'EMA states loaded from step {self.ema.step}')
        self.network.module.load_state_dict(checkpoint['weights'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.log.info(f'Checkpoint loaded from {path} (iteration {it}).')
        return it

    def load_weights(self, path):
        map_location = f'cuda:{local_rank}'
        src_dict = torch.load(path, map_location={'cuda:0': map_location}, weights_only=True)
        self.network.module.load_weights(src_dict)
        self.log.info(f'Network weights loaded from {path}.')

    def weights(self):
        return self.network.module.state_dict()

    def enter_train(self):
        self.integrator = self.train_integrator
        self.network.train()
        return self

    def enter_val(self):
        self.network.eval()
        return self


class RunnerFlowMatching(Runner):
    """Flow matching with logit-normal time sampling and classifier-free guidance."""

    def setup_sampler(self, cfg: DictConfig) -> None:
        self.fm = FlowMatching(cfg.sampling.min_sigma,
                               inference_mode=cfg.sampling.method,
                               num_steps=cfg.sampling.num_steps,
                               reverse_flow=bool(cfg.sampling.get('reverse_flow', True)),
                               shift=float(cfg.sampling.get('shift', 1.0)))
        self.t_mean = cfg.sampling.mean
        self.t_scale = cfg.sampling.scale
        info_if_rank_zero(self.log, f'Flow matching: cfg_strength={cfg.cfg_strength}, '
                          f'steps={cfg.sampling.num_steps}, shift={cfg.sampling.get("shift", 1.0)}')

    def compute_loss(self, text_f, text_f_c, text_padding_mask, x1, loss_mask=None):
        net = self.network.module
        t = log_normal_sample(x1, generator=self.rng, m=self.t_mean, s=self.t_scale)
        t = self.fm._apply_shift(t)
        x0, x1, xt, _ = self.fm.get_x0_xt_c(x1, t, Cs=[], generator=self.rng)

        # classifier-free training: drop the text condition
        null_text = torch.rand(x1.shape[0], device=x1.device, generator=self.rng) < self.null_condition_probability
        text_f = text_f.clone()
        text_f_c = text_f_c.clone()
        text_f[null_text] = net.empty_string_feat
        text_f_c[null_text] = net.empty_string_feat_c
        if text_padding_mask is not None:
            text_padding_mask = text_padding_mask.clone()
            text_padding_mask[null_text] = net._empty_string_padding_mask

        pred_v = self.network(xt, text_f, text_f_c, t, text_padding_mask=text_padding_mask)
        if loss_mask is not None:
            target_v = x0 - x1 if self.fm.reverse_flow else x1 - x0
            per_frame = (pred_v - target_v).pow(2).mean(dim=-1)
            loss = (per_frame * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1)
        else:
            loss = self.fm.loss(pred_v, x0, x1)
        return loss, loss.mean(), t

    def generate(self, captions: list[str]) -> torch.Tensor:
        return generate_fm(captions,
                           feature_utils=self.features,
                           net=self.network.module,
                           fm=self.fm,
                           rng=self.rng,
                           cfg_strength=self.cfg_strength)


class RunnerMeanFlow(Runner):
    """MeanFlow training with guidance-scale sampling (one-step generation)."""

    def setup_sampler(self, cfg: DictConfig) -> None:
        mf_cfg = cfg.meanflow
        self.mf = MeanFlow(null_condition_probability=self.null_condition_probability,
                           flexible_cfg=bool(mf_cfg.flexible_cfg),
                           cfg_scale_max=float(mf_cfg.cfg_scale_max),
                           cfg_scale_beta=float(mf_cfg.cfg_scale_beta),
                           shift_enabled=bool(mf_cfg.shift_enabled),
                           shift=float(mf_cfg.shift))
        # sampler used for generation (same settings as infer.py)
        self.mf_sampler = MeanFlow(steps=int(mf_cfg.get('sampling_steps', 1)))
        info_if_rank_zero(self.log, f'MeanFlow: cfg_strength={cfg.cfg_strength}, flexible_cfg={mf_cfg.flexible_cfg}, '
                          f'cfg_scale_max={mf_cfg.cfg_scale_max}, shift={mf_cfg.shift}')

    def compute_loss(self, text_f, text_f_c, text_padding_mask, x1, loss_mask=None):
        net = self.network.module
        loss, _, t = self.mf.loss(self.network,
                                  x1,
                                  text_f,
                                  text_f_c,
                                  net.empty_string_feat,
                                  net.empty_string_feat_c,
                                  text_padding_mask=text_padding_mask,
                                  empty_string_padding_mask=net._empty_string_padding_mask)
        return loss, loss.mean(), t

    def generate(self, captions: list[str]) -> torch.Tensor:
        return generate_mf(captions,
                           feature_utils=self.features,
                           net=self.network.module,
                           mf=self.mf_sampler,
                           rng=self.rng,
                           cfg_strength=self.cfg_strength,
                           cfg_interval_min=0.0,
                           cfg_interval_max=1.0)


def build_runner(cfg: DictConfig, **kwargs) -> Runner:
    runner_cls = RunnerMeanFlow if cfg.get('use_meanflow', False) else RunnerFlowMatching
    return runner_cls(cfg, **kwargs)
