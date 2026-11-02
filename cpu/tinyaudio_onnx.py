"""TinyAudio-MF on CPU with ONNX Runtime.

ONNX files are downloaded from the Hugging Face Hub on first use.

Usage:
    python tinyaudio_onnx.py "A dog is barking in a room"
    python tinyaudio_onnx.py "Rain falling on a window" --output out/rain.wav
"""
import json
import logging
import os
import time
import wave
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

log = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "standard"
HF_REPO_ID = "junxiliu/TinyAudio"
HF_SUBFOLDER = "cpu/standard"


def _is_complete(model_dir: Path) -> bool:
    pipeline_path = model_dir / "pipeline.json"
    if not pipeline_path.exists():
        return False
    artifacts = json.loads(pipeline_path.read_text(encoding="utf-8"))["artifacts"]
    return all((model_dir / name).exists() for name in artifacts.values())


def resolve_model_dir(model_dir: str | Path | None = None, *, repo_id: str = HF_REPO_ID,
                      revision: str | None = None, local_files_only: bool = False) -> Path:
    """Return a folder with the full ONNX package, downloading it from the Hub if needed."""
    if model_dir is not None:
        model_dir = Path(model_dir).expanduser()
        if model_dir != DEFAULT_MODEL_DIR:
            if not _is_complete(model_dir):
                raise FileNotFoundError(f"Incomplete ONNX package: {model_dir}")
            return model_dir
    if _is_complete(DEFAULT_MODEL_DIR):
        return DEFAULT_MODEL_DIR

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    kwargs = dict(repo_id=repo_id, revision=revision, allow_patterns=[f"{HF_SUBFOLDER}/*"])
    try:
        model_dir = Path(snapshot_download(local_files_only=True, **kwargs)) / HF_SUBFOLDER
        if not _is_complete(model_dir):
            raise LocalEntryNotFoundError(HF_SUBFOLDER)
    except LocalEntryNotFoundError:
        if local_files_only:
            raise FileNotFoundError(f"{HF_SUBFOLDER} is not cached; run once without --offline to download it")
        log.info("Downloading %s from https://huggingface.co/%s", HF_SUBFOLDER, repo_id)
        model_dir = Path(snapshot_download(**kwargs)) / HF_SUBFOLDER
    if not _is_complete(model_dir):
        raise FileNotFoundError(f"{HF_SUBFOLDER} is incomplete in https://huggingface.co/{repo_id}")
    return model_dir


def _make_session(model_path: Path, *, provider: str, num_threads: int,
                  num_interop_threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if num_threads > 0:
        opts.intra_op_num_threads = num_threads
    if num_interop_threads > 0:
        opts.inter_op_num_threads = num_interop_threads

    available = ort.get_available_providers()
    if provider not in available:
        log.debug("Provider %s unavailable (available: %s); using CPUExecutionProvider.", provider, available)
        provider = "CPUExecutionProvider"
    providers = [provider]
    if provider != "CPUExecutionProvider" and "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
    log.debug("Loaded %s | providers=%s", model_path.name, session.get_providers())
    return session


def _save_wav(audio: np.ndarray, path: Path, sample_rate: int) -> None:
    audio = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def _default_output(num_steps: int, cfg_strength: float) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("out") / f"{stamp}_steps{num_steps}_cfg{cfg_strength}.wav"


class _Timer:

    def __init__(self, stage_times: dict[str, list[float]], name: str) -> None:
        self.stage_times = stage_times
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self.start
        self.stage_times.setdefault(self.name, []).append(elapsed)
        log.debug("%s: %.3f s", self.name, elapsed)
        return False


class TinyAudio:
    """ONNX Runtime pipeline: TA-CLAP text encoder -> MeanFlow steps -> TA-VAE decoder."""

    CFG_INTERVAL_MIN = 0.0
    CFG_INTERVAL_MAX = 1.0

    def __init__(self,
                 model_dir: str | Path | None = None,
                 *,
                 repo_id: str = HF_REPO_ID,
                 local_files_only: bool = False,
                 num_threads: int | None = None,
                 num_interop_threads: int | None = None,
                 ort_provider: str | None = None,
                 decoder_ort_provider: str | None = None) -> None:
        self.model_dir = resolve_model_dir(model_dir, repo_id=repo_id, local_files_only=local_files_only)
        num_threads = int(num_threads or os.environ.get("CPU_NUM_THREADS", os.cpu_count() or 4))
        num_interop_threads = int(num_interop_threads or os.environ.get("CPU_NUM_INTEROP_THREADS", "1"))
        ort_provider = ort_provider or os.environ.get("ORT_PROVIDER", "CPUExecutionProvider")
        decoder_ort_provider = decoder_ort_provider or os.environ.get("ORT_DECODER_PROVIDER",
                                                                      "OpenVINOExecutionProvider")
        self.stage_times: dict[str, list[float]] = {}

        pipeline_path = self.model_dir / "pipeline.json"
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Missing pipeline metadata: {pipeline_path}")
        self.meta = json.loads(pipeline_path.read_text(encoding="utf-8"))
        artifacts = self.meta["artifacts"]

        tokenizer_dir = Path(self.meta["tokenizer_dir"])
        if not tokenizer_dir.is_absolute():
            tokenizer_dir = self.model_dir / tokenizer_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)

        session = dict(num_threads=num_threads, num_interop_threads=num_interop_threads)
        self.text_encoder = _make_session(self.model_dir / artifacts["text_encoder"], provider=ort_provider,
                                          **session)
        self.meanflow_step = _make_session(self.model_dir / artifacts["meanflow_step"], provider=ort_provider,
                                           **session)
        self.vae_decode = _make_session(self.model_dir / artifacts["vae_decode"], provider=decoder_ort_provider,
                                        **session)

    def _timer(self, name: str) -> _Timer:
        return _Timer(self.stage_times, name)

    def generate_audio(self, text: str, *, num_steps: int = 1, cfg_strength: float = 7.0,
                       seed: int = 123) -> np.ndarray:
        """Return a mono float32 waveform in [-1, 1]."""
        with self._timer("tokenize"):
            tokens = self.tokenizer([text],
                                    max_length=int(self.meta["prompt_max_length"]),
                                    padding="max_length",
                                    truncation=True,
                                    return_tensors="np")

        with self._timer("text_encode"):
            text_hidden, _, text_padding_mask = self.text_encoder.run(
                ["text_hidden", "text_global", "text_padding_mask"], {
                    "input_ids": tokens["input_ids"].astype(np.int64),
                    "attention_mask": tokens["attention_mask"].astype(np.int64),
                })

        rng = np.random.default_rng(seed)
        x = rng.standard_normal((1, int(self.meta["latent_seq_len"]), int(self.meta["latent_dim"])),
                                dtype=np.float32)

        with self._timer("meanflow_steps"):
            schedule = np.linspace(1.0, 0.0, num_steps + 1, dtype=np.float32)
            for i in range(num_steps):
                x = self.meanflow_step.run(
                    ["x_next"], {
                        "text_hidden": text_hidden.astype(np.float32),
                        "text_padding_mask": text_padding_mask.astype(bool),
                        "x": x.astype(np.float32),
                        "t": schedule[i:i + 1],
                        "r": schedule[i + 1:i + 2],
                        "cfg_strength": np.array([cfg_strength], dtype=np.float32),
                        "cfg_interval_min": np.array([self.CFG_INTERVAL_MIN], dtype=np.float32),
                        "cfg_interval_max": np.array([self.CFG_INTERVAL_MAX], dtype=np.float32),
                    })[0]

        with self._timer("vae_decode"):
            audio = self.vae_decode.run(["audio"], {"x": x.astype(np.float32)})[0]
        return np.asarray(audio, dtype=np.float32)

    def generate(self, text: str, output_path: str | Path | None = None, *, num_steps: int = 1,
                 cfg_strength: float = 7.0, seed: int = 123) -> Path:
        """Generate audio and save it as a 16-bit WAV file."""
        output_path = Path(output_path) if output_path is not None else _default_output(num_steps, cfg_strength)
        audio = self.generate_audio(text, num_steps=num_steps, cfg_strength=cfg_strength, seed=seed)
        _save_wav(audio, output_path, int(self.meta["audio_sample_rate"]))
        return output_path

    def log_stage_summary(self) -> None:
        log.info("Stage timings:")
        for name, values in self.stage_times.items():
            log.info("  %-16s mean=%.3f s  last=%.3f s  n=%d", name, sum(values) / len(values), values[-1],
                     len(values))


def tinyaudio(text: str, output_path: str | Path | None = None, *, model_dir: str | Path | None = None,
              num_steps: int = 1, cfg_strength: float = 7.0, seed: int = 123) -> Path:
    """One-call helper: tinyaudio("prompt", "out.wav")."""
    return TinyAudio(model_dir=model_dir).generate(text, output_path, num_steps=num_steps,
                                                   cfg_strength=cfg_strength, seed=seed)


def main() -> None:
    parser = ArgumentParser(description="TinyAudio-MF CPU inference with ONNX Runtime.")
    parser.add_argument("text", nargs="?", default="A dog is barking in a room")
    parser.add_argument("--model_dir", type=Path, default=None,
                        help="local ONNX package; downloaded from the Hugging Face Hub when omitted")
    parser.add_argument("--repo_id", type=str, default=HF_REPO_ID)
    parser.add_argument("--offline", action="store_true", help="use local or cached files only")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument("--cfg_strength", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_threads", type=int, default=None)
    parser.add_argument("--num_interop_threads", type=int, default=None)
    parser.add_argument("--ort_provider", type=str, default=None)
    parser.add_argument("--decoder_ort_provider", type=str, default=None)
    parser.add_argument("--warmup_runs", type=int, default=0)
    parser.add_argument("--repeat_runs", type=int, default=1)
    parser.add_argument("--verbose", action="store_true", help="print loading details and stage timings")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="[%(levelname)s] %(message)s")
    model = TinyAudio(args.model_dir,
                      repo_id=args.repo_id,
                      local_files_only=args.offline,
                      num_threads=args.num_threads,
                      num_interop_threads=args.num_interop_threads,
                      ort_provider=args.ort_provider,
                      decoder_ort_provider=args.decoder_ort_provider)
    kwargs = dict(num_steps=args.num_steps, cfg_strength=args.cfg_strength)

    for i in range(max(args.warmup_runs, 0)):
        model.generate_audio(args.text, seed=args.seed + i + 1, **kwargs)

    totals = []
    audio = None
    for i in range(max(args.repeat_runs, 1)):
        start = time.perf_counter()
        audio = model.generate_audio(args.text, seed=args.seed + i, **kwargs)
        totals.append(time.perf_counter() - start)

    output_path = args.output or _default_output(args.num_steps, args.cfg_strength)
    _save_wav(audio, output_path, int(model.meta["audio_sample_rate"]))

    if args.verbose or args.repeat_runs > 1:
        log.info("Inference time: mean %.3f s over %d run(s)", sum(totals) / len(totals), len(totals))
        model.log_stage_summary()
    print(output_path)


if __name__ == "__main__":
    main()
