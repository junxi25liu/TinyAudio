import json
import logging
from collections import OrderedDict
from typing import Iterator, List, Optional

import torch
import torch.distributed as dist
import torchaudio
from torch.utils.data import Dataset, Sampler
from torchaudio.transforms import Resample
from tqdm import tqdm

log = logging.getLogger(__name__)


def pad_sequence(data):
    """Right-pad a list of 1D tensors to the same length."""
    length = [x.shape[0] for x in data]
    padded = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
    return padded, length


class AudioCaptionDataset(Dataset):
    """Text-audio pairs decoded on the fly.

    Each dataset entry in the config has a `name`, an integer sampling `weight`, and a list of `meta`
    JSONL files. One JSONL line describes one clip:
        {"audio_id": "...", "audio_path": "/path/to/clip.wav", "caption": "..." or [...], "duration": 9.8}
    Clips with several captions are expanded into one sample per caption.
    """

    def __init__(self,
                 datasets,
                 sample_rate: int = 44100,
                 max_length: int = 10,
                 audio_min_duration: float = 2.0,
                 use_dynamic_batch: bool = False):
        self.sample_rate = sample_rate
        self.max_length = max_length
        self._resamplers: dict[int, Resample] = {}
        self._sr_cache: 'OrderedDict[str, int]' = OrderedDict()
        self._sr_cache_size = 65536
        self._bad_audio_paths: set[str] = set()

        self.data = []
        for dataset in tqdm(datasets, desc='Loading datasets'):
            items = []
            log.info(f"Loading {dataset['name']} ...")
            for meta in dataset['meta']:
                with open(meta, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        duration = item.get('duration', max_length)
                        if duration is None:
                            if use_dynamic_batch:
                                continue
                            duration = max_length
                        if duration < audio_min_duration:
                            continue
                        captions = item.get('caption', [])
                        if isinstance(captions, str):
                            captions = [captions]
                        if not isinstance(captions, list):
                            continue
                        for caption in captions:
                            items.append({
                                'id': item['audio_id'],
                                'audio_path': item.get('audio_path', ''),
                                'caption': caption,
                                'duration': duration,
                            })
            self.data.extend(items * int(dataset['weight']))

    def _get_resampler(self, src_sr: int) -> Optional[Resample]:
        if src_sr == self.sample_rate:
            return None
        if src_sr not in self._resamplers:
            self._resamplers[src_sr] = Resample(orig_freq=src_sr, new_freq=self.sample_rate)
        return self._resamplers[src_sr]

    def _source_sample_rate(self, audio_path: str) -> int:
        sr = self._sr_cache.get(audio_path)
        if sr is None:
            sr = int(torchaudio.info(audio_path).sample_rate)
            self._sr_cache[audio_path] = sr
            if len(self._sr_cache) > self._sr_cache_size:
                self._sr_cache.popitem(last=False)
        return sr

    def _load_waveform(self, audio_path: str) -> Optional[torch.Tensor]:
        if audio_path in self._bad_audio_paths:
            return None
        try:
            src_sr = self._source_sample_rate(audio_path)
            waveform, sr = torchaudio.load(audio_path, num_frames=int(self.max_length * src_sr))
        except Exception:
            self._bad_audio_paths.add(audio_path)
            return None
        waveform = waveform[0]
        resampler = self._get_resampler(sr)
        if resampler is not None:
            waveform = resampler(waveform)
        return waveform

    def __getitem__(self, index):
        item = self.data[index]
        waveform = self._load_waveform(item['audio_path'])
        if waveform is None:
            return None
        return {'id': item['id'], 'waveform': waveform, 'caption': item['caption']}

    def __len__(self):
        return len(self.data)

    def get_frame_len(self, index):
        return self.data[index]['duration']

    def collate_fn(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        waveforms, lengths = pad_sequence([b['waveform'] for b in batch])
        return {
            'id': [b['id'] for b in batch],
            'waveforms': waveforms,
            'waveform_lengths': lengths,
            'caption': [b['caption'] for b in batch],
        }


class TextOnlyDataset(Dataset):
    """Captions for evaluation. JSONL lines: {"id": "...", "caption": "..."}."""

    def __init__(self, jsonl_path: str):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            self.items = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        return {'id': str(item['id']), 'caption': item['caption']}


class DynamicBatchSampler(Sampler[list[int]]):
    """Groups clips of similar duration so that each batch holds at most `frames_threshold` seconds of audio."""

    def __init__(self,
                 sampler: Sampler[int],
                 frames_threshold: float,
                 max_samples: int = 0,
                 random_seed: Optional[int] = None,
                 drop_residual: bool = False):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0

        data_source = self.sampler.data_source
        indices = sorted(((idx, data_source.get_frame_len(idx)) for idx in self.sampler), key=lambda e: e[1])

        batches, batch, batch_frames = [], [], 0
        for idx, frame_len in indices:
            if batch_frames + frame_len <= frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if batch:
                    batches.append(batch)
                if frame_len <= frames_threshold:
                    batch, batch_frames = [idx], frame_len
                else:
                    batch, batch_frames = [], 0
        if not drop_residual and batch:
            batches.append(batch)
        self.batches = batches
        self.drop_last = True

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _shuffled(self) -> list[list[int]]:
        if self.random_seed is None:
            return self.batches
        g = torch.Generator()
        g.manual_seed(self.random_seed + self.epoch)
        return [self.batches[i] for i in torch.randperm(len(self.batches), generator=g).tolist()]

    def __iter__(self):
        return iter(self._shuffled())

    def __len__(self):
        return len(self.batches)


class DistributedDynamicBatchSampler(DynamicBatchSampler):
    """DynamicBatchSampler whose global batches are shuffled identically on all ranks and then split by rank."""

    def __init__(self,
                 sampler: Sampler[int],
                 frames_threshold: float,
                 max_samples: int = 0,
                 random_seed: Optional[int] = None,
                 drop_residual: bool = False,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 drop_last: bool = False):
        self.num_replicas = num_replicas if num_replicas is not None else dist.get_world_size()
        self.rank = rank if rank is not None else dist.get_rank()
        self.drop_last_ddp = drop_last
        super().__init__(sampler, frames_threshold, max_samples, random_seed, drop_residual)
        total = len(self.batches)
        self.num_local_batches = total // self.num_replicas if drop_last else -(-total // self.num_replicas)

    def __iter__(self) -> Iterator[List[int]]:
        local = self._shuffled()[self.rank::self.num_replicas]
        return iter(local[:self.num_local_batches])

    def __len__(self) -> int:
        return self.num_local_batches
