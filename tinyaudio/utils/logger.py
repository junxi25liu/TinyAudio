"""Console + TensorBoard logging."""
import datetime
import logging
import math
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from torch.utils.tensorboard import SummaryWriter

from tinyaudio.utils.time_estimator import PartialTimeEstimator, TimeEstimator


def plot_spectrogram(spectrogram: np.ndarray, title=None, ylabel='freq_bin', ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(spectrogram, origin='lower', aspect='auto', interpolation='nearest')


class TensorboardLogger:

    def __init__(self, exp_id: str, run_dir: Union[Path, str], py_logger: logging.Logger, *, is_rank0: bool = False):
        self.exp_id = exp_id
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.py_log = py_logger
        self.tb_log = SummaryWriter(str(self.run_dir)) if is_rank0 else None

        # used when logging metrics
        self.batch_timer: Optional[TimeEstimator] = None
        self.data_timer: Optional[PartialTimeEstimator] = None

    def log_scalar(self, tag: str, x: float, it: int):
        if self.tb_log is None:
            return
        if math.isnan(x) and 'grad_norm' not in tag:
            self.py_log.warning(f'NaN detected in {tag} at iteration {it}')
        self.tb_log.add_scalar(tag, x, it)

    def log_metrics(self, prefix: str, metrics: dict[str, float], it: int, ignore_timer: bool = False):
        msg = f'{self.exp_id}-{prefix} - it {it:6d}: '
        metrics_msg = ''
        for k, v in sorted(metrics.items()):
            self.log_scalar(f'{prefix}/{k}', v, it)
            metrics_msg += f'{k: >10}:{v:.7f},\t'

        if self.batch_timer is not None and not ignore_timer:
            self.batch_timer.update()
            avg_time = self.batch_timer.get_and_reset_avg_time()
            data_time = self.data_timer.get_and_reset_avg_time()
            self.log_scalar(f'{prefix}/avg_time', avg_time, it)
            self.log_scalar(f'{prefix}/data_time', data_time, it)

            est = datetime.timedelta(seconds=self.batch_timer.get_est_remaining(it))
            if est.days > 0:
                remaining_str = f'{est.days}d {est.seconds // 3600}h'
            else:
                remaining_str = f'{est.seconds // 3600}h {(est.seconds % 3600) // 60}m'
            eta_str = (datetime.datetime.now().astimezone() + est).strftime('%Y-%m-%d %H:%M:%S %Z')
            msg = f'{msg} avg_time:{avg_time:.3f},data:{data_time:.3f},remaining:{remaining_str},eta:{eta_str},\t'

        self.py_log.info(f'{msg} {metrics_msg}')

    def log_histogram(self, tag: str, hist: torch.Tensor, it: int):
        if self.tb_log is None:
            return
        hist = hist.cpu().numpy()
        fig, ax = plt.subplots()
        x_range = np.linspace(0, 1, len(hist))
        ax.bar(x_range, hist, width=1 / (len(hist) - 1))
        ax.set_xticks(x_range)
        ax.set_xticklabels(x_range)
        plt.tight_layout()
        self.tb_log.add_figure(tag, fig, it)
        plt.close()

    def log_audio(self,
                  prefix: str,
                  tag: str,
                  waveform: torch.Tensor,
                  it: Optional[int] = None,
                  *,
                  subdir: Optional[Path] = None,
                  sample_rate: int = 44100) -> Path:
        audio_dir = self.run_dir / prefix if subdir is None else self.run_dir / subdir / prefix
        audio_dir.mkdir(exist_ok=True, parents=True)
        name = f'{tag}.flac' if it is None else f'{it:09d}_{tag}.flac'
        torchaudio.save(audio_dir / name, waveform.cpu().float(), sample_rate=sample_rate, channels_first=True)
        return Path(audio_dir)

    def log_spectrogram(self, prefix: str, tag: str, spec: torch.Tensor, it: Optional[int], *,
                        subdir: Optional[Path] = None):
        spec_dir = self.run_dir / prefix if subdir is None else self.run_dir / subdir / prefix
        spec_dir.mkdir(exist_ok=True, parents=True)
        name = f'{tag}.png' if it is None else f'{it:09d}_{tag}.png'
        plot_spectrogram(spec.cpu().float())
        plt.tight_layout()
        plt.savefig(spec_dir / name)
        plt.close()

    def log_string(self, tag: str, x: str):
        self.py_log.info(f'{tag} - {x}')
        if self.tb_log is not None:
            self.tb_log.add_text(tag, x)

    def debug(self, x):
        self.py_log.debug(x)

    def info(self, x):
        self.py_log.info(x)

    def warning(self, x):
        self.py_log.warning(x)

    def error(self, x):
        self.py_log.error(x)

    def critical(self, x):
        self.py_log.critical(x)

    def complete(self):
        if self.tb_log is not None:
            self.tb_log.flush()
