"""Saved predictions, accuracy at clip and recording level, and participant-level bootstrap intervals."""

import json
import os

import numpy as np
import pandas as pd


def save_predictions(folder, name, proba, val_accuracy, train_accuracy, info):
    """Write the predictions of one experiment to `<folder>/<name>.npz`.

    proba           (seeds, clips, classes) probabilities of every clip, each predicted by the model of
                    the fold in which the clip belongs to the test set (float16)
    val_accuracy    (seeds, folds) clip accuracy on the validation fold
    train_accuracy  (seeds, folds) clip accuracy on the training folds
    info            settings and per-run details, stored as JSON text
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name + ".npz")
    np.savez_compressed(path, proba=np.asarray(proba, dtype=np.float16),
                        val_accuracy=np.asarray(val_accuracy, dtype=np.float32),
                        train_accuracy=np.asarray(train_accuracy, dtype=np.float32),
                        info=np.asarray(json.dumps(info, sort_keys=True)))
    return path


def load_predictions(path):
    with np.load(path) as data:
        return {"proba": data["proba"].astype(np.float32), "val_accuracy": data["val_accuracy"],
                "train_accuracy": data["train_accuracy"], "info": json.loads(str(data["info"]))}


def load_clips(path):
    """The clip table: one row per 10 s clip with its recording, participant, kind, song, label and fold."""
    return pd.read_csv(path)


def recording_proba(proba, recording_index, n_recordings=None):
    """Mean class probabilities over the clips of each recording. `proba` is (..., clips, classes)."""
    proba = np.asarray(proba, dtype=np.float64)
    recording_index = np.asarray(recording_index)
    n_recordings = int(recording_index.max()) + 1 if n_recordings is None else n_recordings
    flat = proba.reshape((-1,) + proba.shape[-2:])
    out = np.zeros((flat.shape[0], n_recordings, proba.shape[-1]), dtype=np.float64)
    for row in range(flat.shape[0]):
        np.add.at(out[row], recording_index, flat[row])
    counts = np.bincount(recording_index, minlength=n_recordings)
    return (out / counts[:, None]).reshape(proba.shape[:-2] + (n_recordings, proba.shape[-1]))


def bootstrap_interval(correct, groups, n_boot=2000, seed=0, level=0.95):
    """Percentile interval of an accuracy when whole groups (participants) are resampled.

    `correct` holds one value per item (0 or 1, or the share of seeds that were right). Items of one
    participant are not independent, so the participants are drawn with replacement, not the items.
    """
    correct = np.asarray(correct, dtype=np.float64)
    codes = pd.factorize(np.asarray(groups))[0]
    n_groups = codes.max() + 1
    sums = np.bincount(codes, weights=correct, minlength=n_groups)
    counts = np.bincount(codes, minlength=n_groups).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_groups, size=(n_boot, n_groups))
    stats = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def bootstrap_difference(correct_a, correct_b, groups, n_boot=2000, seed=0, level=0.95):
    """Interval of accuracy(a) minus accuracy(b) on the same items, resampling participants."""
    diff = np.asarray(correct_a, dtype=np.float64) - np.asarray(correct_b, dtype=np.float64)
    codes = pd.factorize(np.asarray(groups))[0]
    n_groups = codes.max() + 1
    sums = np.bincount(codes, weights=diff, minlength=n_groups)
    counts = np.bincount(codes, minlength=n_groups).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_groups, size=(n_boot, n_groups))
    stats = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(diff.mean()), float(lo), float(hi)


def summarise(proba, clips, n_boot=2000):
    """Accuracy summary of one experiment.

    Returns a dict with, for clips and for recordings: the accuracy of every seed, their mean and
    standard deviation (population formula, 0 for a single seed), the 95% participant-bootstrap interval
    of the seed-averaged accuracy, and the mean accuracy per test fold.
    """
    labels = clips["label"].to_numpy()
    out = {}
    recording_index = pd.factorize(clips["recording"], sort=True)[0]   # dense, in case a recording gave no clip
    first = clips.drop_duplicates("recording").sort_values("recording")
    levels = {
        "clip": (proba.argmax(axis=2) == labels[None, :], clips["participant"].to_numpy(),
                 clips["fold"].to_numpy()),
        "recording": (recording_proba(proba, recording_index).argmax(axis=2) == first["label"].to_numpy()[None, :],
                      first["participant"].to_numpy(), first["fold"].to_numpy()),
    }
    for level, (correct, participants, fold) in levels.items():
        per_seed = correct.mean(axis=1)
        share = correct.mean(axis=0)
        lo, hi = bootstrap_interval(share, participants, n_boot=n_boot)
        out[level] = {
            "n": int(correct.shape[1]),
            "per_seed": [float(v) for v in per_seed],
            "mean": float(per_seed.mean()),
            "std": float(per_seed.std()),
            "ci95": [lo, hi],
            "per_fold": [float(share[fold == k].mean()) for k in sorted(set(fold.tolist()))],
        }
    return out
