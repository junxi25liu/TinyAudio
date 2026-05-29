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
MODE_DIRS = {
    "standard": Path(__file__).resolve().parent / "standard",
}
AUTO_PROVIDER = "auto"
AUTO_PROVIDER_ORDER = (
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)


def _resolve_provider(provider: str | None) -> str:
    requested = provider or AUTO_PROVIDER
    available = ort.get_available_providers()
    if requested.lower() == AUTO_PROVIDER:
        for candidate in AUTO_PROVIDER_ORDER:
            if candidate in available:
                return candidate
        return "CPUExecutionProvider"
    if requested not in available:
        log.warning(
            "Provider %s unavailable; available=%s. Falling back to CPUExecutionProvider.",
            requested,
            available,
        )
        return "CPUExecutionProvider"
    return requested


def _make_session(
    model_path: Path,
    *,
    provider: str,
    num_threads: int,
    num_interop_threads: int,
) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if num_threads > 0:
        opts.intra_op_num_threads = num_threads
    if num_interop_threads > 0:
        opts.inter_op_num_threads = num_interop_threads

    provider = _resolve_provider(provider)
    providers = [provider]
    if provider != "CPUExecutionProvider" and "CPUExecutionProvider" in ort.get_available_providers():
        providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
    log.debug("Loaded %s | providers=%s", model_path.name, session.get_providers())
    return session


def _filter_feed(session: ort.InferenceSession, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = {inp.name for inp in session.get_inputs()}
    return {key: value for key, value in feed.items() if key in names}


def _save_wav(audio: np.ndarray, path: Path, sample_rate: int) -> None:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_f:
        wav_f.setnchannels(1)
        wav_f.setsampwidth(2)
        wav_f.setframerate(sample_rate)
        wav_f.writeframes(pcm.tobytes())


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
    """CPU ONNX inference wrapper for TinyAudio."""

    CFG_INTERVAL_MIN = 0.0
    CFG_INTERVAL_MAX = 1.0

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        *,
        num_threads: int | None = None,
        num_interop_threads: int | None = None,
        ort_provider: str | None = None,
        decoder_ort_provider: str | None = None,
        front_precision: str = "fp32",
    ) -> None:
        self.model_dir = Path(model_dir).expanduser()
        self.num_threads = int(num_threads or os.environ.get("CPU_NUM_THREADS", "16"))
        self.num_interop_threads = int(num_interop_threads or os.environ.get("CPU_NUM_INTEROP_THREADS", "1"))
        self.ort_provider = ort_provider or os.environ.get("ORT_PROVIDER", AUTO_PROVIDER)
        self.decoder_ort_provider = decoder_ort_provider or os.environ.get("ORT_DECODER_PROVIDER", AUTO_PROVIDER)
        self.front_precision = "fp32" if front_precision == "bf32" else front_precision
        if self.front_precision not in {"int8", "fp32"}:
            raise ValueError("front_precision must be one of: int8, fp32, bf32")
        self.stage_times: dict[str, list[float]] = {}

        pipeline_path = self.model_dir / "pipeline.json"
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Missing pipeline metadata: {pipeline_path}")
        self.meta = json.loads(pipeline_path.read_text(encoding="utf-8"))
        artifacts = self.meta["artifacts"]
        use_int8_front = self.front_precision == "int8"
        text_encoder_name = artifacts.get("text_encoder_int8") if use_int8_front else artifacts.get("text_encoder")
        meanflow_step_name = artifacts.get("meanflow_step_int8") if use_int8_front else artifacts.get("meanflow_step")
        text_encoder_name = text_encoder_name or artifacts.get("text_encoder")
        meanflow_step_name = meanflow_step_name or artifacts.get("meanflow_step")
        if not text_encoder_name or not meanflow_step_name:
            raise ValueError(
                "Selected front_precision requires text_encoder/text_encoder_int8 and "
                "meanflow_step/meanflow_step_int8 artifacts."
            )
        semanticvae_decode_name = artifacts["semanticvae_decode"]
        identity_vocode_name = artifacts.get("bigvgan_vocode")

        tokenizer_dir = Path(self.meta["tokenizer_dir"])
        if not tokenizer_dir.is_absolute():
            tokenizer_dir = self.model_dir / tokenizer_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        self.text_encoder = _make_session(
            self.model_dir / text_encoder_name,
            provider=self.ort_provider,
            num_threads=self.num_threads,
            num_interop_threads=self.num_interop_threads,
        )
        self.meanflow_step = _make_session(
            self.model_dir / meanflow_step_name,
            provider=self.ort_provider,
            num_threads=self.num_threads,
            num_interop_threads=self.num_interop_threads,
        )
        self.semanticvae_decode = _make_session(
            self.model_dir / semanticvae_decode_name,
            provider=self.decoder_ort_provider,
            num_threads=self.num_threads,
            num_interop_threads=self.num_interop_threads,
        )
        self.identity_vocode = None
        if identity_vocode_name:
            self.identity_vocode = _make_session(
                self.model_dir / identity_vocode_name,
                provider=self.decoder_ort_provider,
                num_threads=self.num_threads,
                num_interop_threads=self.num_interop_threads,
            )
        self.providers = {
            "text": self.text_encoder.get_providers()[0],
            "meanflow": self.meanflow_step.get_providers()[0],
            "decoder": self.semanticvae_decode.get_providers()[0],
        }
        if self.identity_vocode is not None:
            self.providers["vocode"] = self.identity_vocode.get_providers()[0]

    def provider_summary(self) -> str:
        return ", ".join(f"{name}={provider}" for name, provider in self.providers.items())

    def _timer(self, name: str) -> _Timer:
        return _Timer(self.stage_times, name)

    def generate_audio(
        self,
        text: str,
        *,
        num_steps: int = 4,
        cfg_strength: float = 7.0,
        seed: int = 123,
    ) -> np.ndarray:
        """Return generated mono audio as float32 waveform in [-1, 1]."""
        with self._timer("tokenize"):
            tokens = self.tokenizer(
                [text],
                max_length=int(self.meta["prompt_max_length"]),
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
            input_ids = tokens["input_ids"].astype(np.int64, copy=False)
            attention_mask = tokens["attention_mask"].astype(np.int64, copy=False)

        with self._timer("text_encode"):
            text_hidden, text_global, text_padding_mask = self.text_encoder.run(
                ["text_hidden", "text_global", "text_padding_mask"],
                {"input_ids": input_ids, "attention_mask": attention_mask},
            )

        with self._timer("sample_latent"):
            rng = np.random.default_rng(seed)
            x = rng.standard_normal(
                (1, int(self.meta["latent_seq_len"]), int(self.meta["latent_dim"])),
                dtype=np.float32,
            )

        with self._timer("meanflow_step_loop"):
            schedule = np.linspace(1.0, 0.0, num_steps + 1, dtype=np.float32)
            for idx in range(num_steps):
                x = self.meanflow_step.run(
                    ["x_next"],
                    _filter_feed(
                        self.meanflow_step,
                        {
                            "text_hidden": text_hidden.astype(np.float32, copy=False),
                            "text_global": text_global.astype(np.float32, copy=False),
                            "text_padding_mask": text_padding_mask.astype(bool, copy=False),
                            "x": x.astype(np.float32, copy=False),
                            "t": np.array([schedule[idx]], dtype=np.float32),
                            "r": np.array([schedule[idx + 1]], dtype=np.float32),
                            "cfg_strength": np.array([cfg_strength], dtype=np.float32),
                            "cfg_interval_min": np.array([self.CFG_INTERVAL_MIN], dtype=np.float32),
                            "cfg_interval_max": np.array([self.CFG_INTERVAL_MAX], dtype=np.float32),
                        },
                    ),
                )[0]

        with self._timer("semanticvae_decode_waveform"):
            audio = self.semanticvae_decode.run(
                ["spec"],
                _filter_feed(self.semanticvae_decode, {"x": x.astype(np.float32, copy=False)}),
            )[0]

        if self.identity_vocode is not None:
            with self._timer("semanticvae_identity_vocode"):
                audio = self.identity_vocode.run(
                    ["audio"],
                    _filter_feed(self.identity_vocode, {"spec": audio.astype(np.float32, copy=False)}),
                )[0]

        return np.asarray(audio, dtype=np.float32)

    def generate(
        self,
        text: str,
        output_path: str | Path | None = None,
        *,
        num_steps: int = 4,
        cfg_strength: float = 7.0,
        seed: int = 123,
    ) -> Path:
        """Generate audio and save it to a WAV file."""
        if output_path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(__file__).resolve().parent / "out" / f"{stamp}_steps{num_steps}_cfg{cfg_strength}.wav"
        output_path = Path(output_path)
        audio = self.generate_audio(text, num_steps=num_steps, cfg_strength=cfg_strength, seed=seed)
        with self._timer("save_wav"):
            _save_wav(audio, output_path, int(self.meta["audio_sample_rate"]))
        return output_path

    def log_stage_summary(self) -> None:
        log.info("Stage timings:")
        for name, values in self.stage_times.items():
            log.info("  %-30s mean=%.3f s last=%.3f s n=%d", name, sum(values) / len(values), values[-1], len(values))


def tinyaudio(
    text: str,
    output_path: str | Path | None = None,
    *,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    num_steps: int = 4,
    cfg_strength: float = 7.0,
    seed: int = 123,
    front_precision: str = "fp32",
) -> Path:
    """Single-call helper: tinyaudio("prompt", "out.wav")."""
    return TinyAudio(model_dir=model_dir, front_precision=front_precision).generate(
        text,
        output_path,
        num_steps=num_steps,
        cfg_strength=cfg_strength,
        seed=seed,
    )


def _parse_args():
    parser = ArgumentParser(description="TinyAudio CPU ONNX inference.")
    parser.add_argument("text", nargs="?", default="A dog is barking in a room")
    parser.add_argument(
        "--mode",
        choices=["standard"],
        default="standard",
        help="Bundled model package to use.",
    )
    parser.add_argument("--model_dir", type=Path, default=None, help="Override bundled model directory.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--cfg_strength", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_threads", type=int, default=int(os.environ.get("CPU_NUM_THREADS", "16")))
    parser.add_argument("--num_interop_threads", type=int, default=int(os.environ.get("CPU_NUM_INTEROP_THREADS", "1")))
    parser.add_argument(
        "--ort_provider",
        type=str,
        default=os.environ.get("ORT_PROVIDER", AUTO_PROVIDER),
        help="ONNX Runtime provider for text/MeanFlow. Use 'auto' for CUDA -> CPU.",
    )
    parser.add_argument(
        "--decoder_ort_provider",
        type=str,
        default=os.environ.get("ORT_DECODER_PROVIDER", AUTO_PROVIDER),
        help="ONNX Runtime provider for waveform decoding. Use 'auto' for CUDA -> CPU.",
    )
    parser.add_argument(
        "--front_precision",
        choices=["int8", "fp32", "bf32"],
        default=os.environ.get("TINY_AUDIO_FRONT_PRECISION", "fp32"),
        help="Precision for text_encoder and meanflow_step. bf32 is accepted as an alias for fp32 ONNX.",
    )
    parser.add_argument("--warmup_runs", type=int, default=0)
    parser.add_argument("--repeat_runs", type=int, default=1)
    parser.add_argument("--verbose", action="store_true", help="Print model loading details and stage timings.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="[%(levelname)s] %(message)s")
    model_dir = args.model_dir or MODE_DIRS[args.mode]
    front_precision = args.front_precision
    if args.model_dir is None and args.mode == "standard" and front_precision == "int8":
        front_precision = "fp32"

    model = TinyAudio(
        model_dir,
        num_threads=args.num_threads,
        num_interop_threads=args.num_interop_threads,
        ort_provider=args.ort_provider,
        decoder_ort_provider=args.decoder_ort_provider,
        front_precision=front_precision,
    )
    log.info("Providers: %s", model.provider_summary())

    for idx in range(max(args.warmup_runs, 0)):
        log.info("Warmup %d/%d", idx + 1, args.warmup_runs)
        model.generate_audio(
            args.text,
            num_steps=args.num_steps,
            cfg_strength=args.cfg_strength,
            seed=args.seed + idx + 1,
        )

    output_path = args.output
    if output_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(__file__).resolve().parent / "out" / f"{stamp}_steps{args.num_steps}_cfg{args.cfg_strength}.wav"

    output = None
    totals = []
    for idx in range(max(args.repeat_runs, 1)):
        start = time.perf_counter()
        output = model.generate_audio(
            args.text,
            num_steps=args.num_steps,
            cfg_strength=args.cfg_strength,
            seed=args.seed + idx,
        )
        elapsed = time.perf_counter() - start
        totals.append(elapsed)
        if args.verbose or args.repeat_runs > 1:
            log.info("Run %d/%d: %.3f s", idx + 1, max(args.repeat_runs, 1), elapsed)

    with model._timer("save_wav"):
        _save_wav(output, output_path, int(model.meta["audio_sample_rate"]))

    if args.verbose or args.repeat_runs > 1:
        log.info("Inference time mean: %.3f s", sum(totals) / len(totals))
        model.log_stage_summary()
    if args.verbose or args.repeat_runs > 1:
        log.info("Saved audio to %s", output_path)
    print(output_path)


if __name__ == "__main__":
    main()
