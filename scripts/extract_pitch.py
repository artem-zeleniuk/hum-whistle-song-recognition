"""Estimate the pitch contour of every clip with probabilistic YIN.

    python scripts/extract_pitch.py --data-dir data/audio

Reads clips.csv from the cache folder (written by extract_features.py) and writes f0.npy (clips, 431):
the fundamental frequency per frame in Hz, NaN where no pitch was found. If f0.npy is already there
the tracking is skipped. Also writes results/pitch_summary.json (how much of the hummed and of the
whistled clips is tracked, and in which frequency range).
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from _common import add_common_arguments, log

from humrec import audio, data, features


def process_recording(path, expected, fmax):
    clips = audio.split_into_clips(audio.load_audio(path))
    if len(clips) != expected:
        raise RuntimeError(f"{path}: {len(clips)} clips, expected {expected}")
    if not clips:
        return np.zeros((0, 431))
    return np.stack([features.pitch_track(clip, fmax=fmax) for _, clip, _ in clips])


def summarise(f0, clips, fmax=features.PITCH_FMAX):
    """Share of tracked frames and frequency range, separately for hums and whistles."""
    out = {"fmin_hz": round(float(features.PITCH_FMIN), 1), "fmax_hz": round(float(fmax), 1)}
    real = np.ceil(clips["real_seconds"].to_numpy() / 10.0 * f0.shape[1]).astype(int)
    for kind in ("hum", "whistle"):
        rows = np.flatnonzero((clips["kind"] == kind).to_numpy())
        voiced = np.array([np.isfinite(f0[i, :real[i]]).mean() for i in rows])
        values = f0[rows][np.isfinite(f0[rows])]
        median = np.array([np.nanmedian(f0[i]) for i in rows])
        out[kind] = {
            "clips": int(len(rows)), "voiced_share": float(voiced.mean()),
            "clips_under_30_percent_voiced": float(np.mean(voiced < 0.3)),
            "f0_percentiles_hz": {str(q): float(np.percentile(values, q)) for q in (5, 25, 50, 75, 95, 99)},
            "voiced_frames_within_10_percent_of_fmax": float(np.mean(values > 0.9 * fmax)),
            "clips_with_median_within_15_percent_of_fmax": float(np.mean(median > 0.85 * fmax)),
            "voiced_frames_above_2093_hz": float(np.mean(values > features.PITCH_FMAX)),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--batch", type=int, default=160)
    parser.add_argument("--wide", action="store_true",
                        help="search up to 4186 Hz instead of 2093 Hz, write f0_wide.npy and pitch_summary_wide.json")
    add_common_arguments(parser)
    args = parser.parse_args()
    clips = pd.read_csv(os.path.join(args.cache_dir, "clips.csv"))
    suffix = "_wide" if args.wide else ""
    fmax = features.PITCH_FMAX_WIDE if args.wide else features.PITCH_FMAX
    target = os.path.join(args.cache_dir, f"f0{suffix}.npy")
    if os.path.exists(target):
        f0 = np.load(target)
        log(f"f0{suffix}.npy is already in the cache, shape {f0.shape}")
        write_summary(f0, clips, args.results_dir, fmax, suffix)
        return
    counts = clips.groupby("file", sort=False).size()
    paths = {os.path.basename(p): p for p in data.find_wav_files(args.data_dir)}
    parts_dir = os.path.join(args.cache_dir, f"parts_f0{suffix}")
    os.makedirs(parts_dir, exist_ok=True)
    files = list(counts.index)
    log(f"{len(clips)} clips of {len(files)} recordings, {args.workers} worker processes")
    features.warm_up()
    tracks = []
    for number, start in enumerate(range(0, len(files), args.batch)):
        part = os.path.join(parts_dir, f"part_{number:03d}.npy")
        if not os.path.exists(part):
            batch = files[start:start + args.batch]
            done = Parallel(n_jobs=args.workers)(
                delayed(process_recording)(paths[f], int(counts[f]), fmax) for f in batch)
            np.save(part + ".tmp.npy", np.concatenate(done).astype(np.float32))
            os.replace(part + ".tmp.npy", part)
            log(f"batch {number + 1} saved")
        tracks.append(np.load(part))
    f0 = np.concatenate(tracks)
    assert len(f0) == len(clips), (len(f0), len(clips))
    np.save(target, f0)
    voiced = np.isfinite(f0)
    log(f"f0{suffix}.npy written, share of voiced frames {voiced.mean():.3f}, "
        f"clips without any voiced frame: {int((~voiced.any(axis=1)).sum())}")
    write_summary(f0, clips, args.results_dir, fmax, suffix)


def write_summary(f0, clips, results_dir, fmax=features.PITCH_FMAX, suffix=""):
    os.makedirs(results_dir, exist_ok=True)
    summary = summarise(f0, clips, fmax)
    with open(os.path.join(results_dir, f"pitch_summary{suffix}.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
        handle.write("\n")
    for kind in ("hum", "whistle"):
        log(f"{kind}: voiced share {summary[kind]['voiced_share']:.3f}, median f0 "
            f"{summary[kind]['f0_percentiles_hz']['50']:.0f} Hz")


if __name__ == "__main__":
    main()
