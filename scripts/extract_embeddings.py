"""Embed every clip with frozen pretrained audio encoders, one embedding per layer.

    python scripts/extract_embeddings.py --data-dir data/audio                  # all encoders
    python scripts/extract_embeddings.py --data-dir data/audio --encoders hubert whisper

First writes waves.npy to the cache folder (every clip as 16-bit samples at 22050 Hz, about 1.3 GB), then
for every encoder emb_<name>.npy with shape (clips, layers, dimensions): the mean over time of the
output of every layer. The encoders are listed in humrec/pretrained.py.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from _common import add_common_arguments, log

from humrec import audio, config, data, pretrained


def clips_of(path, expected):
    clips = audio.split_into_clips(audio.load_audio(path))
    if len(clips) != expected:
        raise RuntimeError(f"{path}: {len(clips)} clips, expected {expected}")
    return np.stack([pretrained.to_int16(clip) for _, clip, _ in clips])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--encoders", nargs="*", default=list(pretrained.ENCODERS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_common_arguments(parser)
    args = parser.parse_args()
    clips = pd.read_csv(os.path.join(args.cache_dir, "clips.csv"))
    waves_path = os.path.join(args.cache_dir, "waves.npy")
    if not os.path.exists(waves_path):
        counts = clips.groupby("file", sort=False).size()
        paths = {os.path.basename(p): p for p in data.find_wav_files(args.data_dir)}
        done = Parallel(n_jobs=args.workers)(delayed(clips_of)(paths[f], int(counts[f])) for f in counts.index)
        waves = np.concatenate(done)
        assert waves.shape == (len(clips), config.CLIP_SAMPLES), waves.shape
        np.save(waves_path, waves)
        log(f"waves.npy written, shape {waves.shape}")
    waves = np.load(waves_path, mmap_mode="r")
    real = clips["real_seconds"].to_numpy()

    for name in args.encoders:
        target = os.path.join(args.cache_dir, f"emb_{name}.npy")
        if os.path.exists(target):
            log(f"{name}: already in the cache")
            continue
        try:
            model, processor = pretrained.load_encoder(name, args.device)
        except Exception as error:  # one encoder that cannot be loaded must not stop the others
            log(f"{name}: could not be loaded ({type(error).__name__}: {error}), skipped")
            continue
        parts = []
        for start in range(0, len(clips), args.batch_size):
            stop = start + args.batch_size
            with torch.autocast(device_type="cuda", enabled=args.device.startswith("cuda")):
                parts.append(pretrained.embed(name, model, processor, waves[start:stop], real[start:stop], args.device))
        embeddings = np.concatenate(parts).astype(np.float32)
        assert len(embeddings) == len(clips) and np.isfinite(embeddings).all(), embeddings.shape
        np.save(target, embeddings.astype(np.float16))
        log(f"{name}: emb_{name}.npy written, shape {embeddings.shape} ({pretrained.ENCODERS[name]['model']})")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
