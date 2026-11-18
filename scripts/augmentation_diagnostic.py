"""How much do noise, pitch shift and time stretch change the MFCC and chroma features of a recording?

    python scripts/augmentation_diagnostic.py --data-dir data/audio

Four recordings per song are transformed with eleven settings. For each setting the script reports
  1. the Pearson correlation between the frame-level MFCC (and chroma) of the original and of the
     transformed recording, averaged over the recordings, and
  2. the ratio of the mean distance between recordings of different songs to the mean distance between
     recordings of the same song, measured on the time-averaged MFCC and chroma (25 numbers per recording).
Writes results/augmentation_diagnostic.csv.
"""

import argparse
import os

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist

from _common import add_common_arguments, log

from humrec import audio, config, data

SETTINGS = [("original", None, 0.0)]
SETTINGS += [(f"noise {a:g}", "noise", a) for a in (1e-4, 5e-4, 1e-3, 2e-3, 5e-3)]
SETTINGS += [(f"pitch +{s:g} semitones", "pitch", s) for s in (0.5, 1.0, 2.0)]
SETTINGS += [(f"stretch x{r:g}", "stretch", r) for r in (1.02, 1.05, 1.10)]


def describe(samples):
    mfcc = librosa.feature.mfcc(y=samples, sr=config.SAMPLE_RATE, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=samples, sr=config.SAMPLE_RATE)
    return mfcc, chroma


def process_recording(path, seed):
    samples = audio.load_audio(path)
    rng = np.random.default_rng(seed)
    reference = describe(samples)
    rows = []
    for name, kind, size in SETTINGS:
        if kind is None:
            changed = samples
        elif kind == "noise":
            changed = audio.add_noise(samples, size, rng)
        elif kind == "pitch":
            changed = audio.shift_pitch(samples, size)
        else:
            changed = audio.stretch_time(samples, size)
        mfcc, chroma = describe(changed)
        rows.append({
            "setting": name,
            "mfcc_correlation": float(np.corrcoef(reference[0].ravel(), mfcc.ravel())[0, 1]),
            "chroma_correlation": float(np.corrcoef(reference[1].ravel(), chroma.ravel())[0, 1]),
            "vector": np.concatenate([mfcc.mean(axis=1), chroma.mean(axis=1)]),
        })
    return rows


def separability(vectors, labels):
    """Mean distance between songs divided by the mean distance within a song."""
    within, between = [], []
    for label in np.unique(labels):
        same = vectors[labels == label]
        other = vectors[labels != label]
        if len(same) > 1:
            within.extend(cdist(same, same)[np.triu_indices(len(same), k=1)])
        between.extend(cdist(same, other).ravel())
    return float(np.mean(between) / np.mean(within))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--per-song", type=int, default=4)
    add_common_arguments(parser)
    args = parser.parse_args()
    recordings = data.list_recordings(args.data_dir)
    sample = recordings.groupby("song", sort=False).sample(n=args.per_song, random_state=0).reset_index(drop=True)
    log(f"{len(sample)} recordings, {len(SETTINGS)} settings")
    done = Parallel(n_jobs=min(args.workers, len(sample)))(
        delayed(process_recording)(row.path, index) for index, row in enumerate(sample.itertuples()))
    labels = sample["label"].to_numpy()
    table = []
    for position, (name, _, _) in enumerate(SETTINGS):
        rows = [recording[position] for recording in done]
        table.append({
            "setting": name,
            "mfcc_correlation": np.mean([r["mfcc_correlation"] for r in rows]),
            "chroma_correlation": np.mean([r["chroma_correlation"] for r in rows]),
            "separability_ratio": separability(np.stack([r["vector"] for r in rows]), labels),
            "mean_vector_norm": float(np.mean([np.linalg.norm(r["vector"]) for r in rows])),
        })
    table = pd.DataFrame(table)
    table.insert(1, "recordings", len(sample))
    os.makedirs(args.results_dir, exist_ok=True)
    table.to_csv(os.path.join(args.results_dir, "augmentation_diagnostic.csv"), index=False, float_format="%.4f")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
