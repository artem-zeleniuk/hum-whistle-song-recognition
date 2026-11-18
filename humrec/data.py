"""Recording table, participant-disjoint folds and the check of the downloaded audio."""

import csv
import hashlib
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config

_NAME = re.compile(r"^(S\d+)_(hum|whistle)_(\d+)_([A-Za-z]+)\.wav$")


def parse_filename(name):
    """Split `S105_whistle_2_Happy.wav` into participant, kind, take and song."""
    match = _NAME.match(os.path.basename(name))
    if match is None:
        raise ValueError(f"unexpected file name: {name}")
    participant, kind, take, song = match.groups()
    if song not in config.SONGS:
        raise ValueError(f"unknown song in file name: {name}")
    return {"participant": participant, "kind": kind, "take": int(take), "song": song}


def find_wav_files(data_dir):
    """All WAV files below `data_dir`, sorted by file name."""
    found = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            if name.lower().endswith(".wav") and not name.startswith("._"):
                found.append(os.path.join(root, name))
    return sorted(found, key=os.path.basename)


def list_recordings(data_dir):
    """One row per recording: file, path, participant, kind, take, song, label."""
    rows = []
    for path in find_wav_files(data_dir):
        row = parse_filename(path)
        row["file"] = os.path.basename(path)
        row["path"] = path
        row["label"] = config.SONGS.index(row["song"])
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no WAV files below {data_dir}")
    table = pd.DataFrame(rows)
    return table[["file", "path", "participant", "kind", "take", "song", "label"]]


def assign_folds(recordings, n_folds=config.N_FOLDS, seed=config.FOLD_SEED):
    """Fold index of every recording.

    All recordings of one participant share a fold, so a model is never tested on a voice it was trained on.
    The folds are stratified by song.
    """
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(recordings), -1, dtype=np.int64)
    parts = splitter.split(recordings, recordings["label"], groups=recordings["participant"])
    for k, (_, test_index) in enumerate(parts):
        fold[test_index] = k
    return fold


def assign_random_folds(labels, n_folds=config.N_FOLDS, seed=config.FOLD_SEED):
    """Fold index per item when the items are split at random within each song, ignoring the participants.

    Used only for the leakage checks: with such folds the same person (for recordings) or even the same
    recording (for clips) appears on both sides of the split.
    """
    from sklearn.model_selection import StratifiedKFold
    labels = np.asarray(labels)
    fold = np.full(len(labels), -1, dtype=np.int64)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test_index) in enumerate(splitter.split(labels, labels)):
        fold[test_index] = k
    return fold


def fold_roles(fold, k, n_folds=config.N_FOLDS):
    """Boolean masks (train, validation, test) for test fold `k`.

    Fold k is the test set, the next fold is the validation set (early stopping and every choice of a
    setting), the remaining folds are the training set.
    """
    fold = np.asarray(fold)
    test = fold == k
    val = fold == (k + 1) % n_folds
    train = ~(test | val)
    return train, val, test


def sha256_of(path, block=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path):
    """`data/manifest.csv`: file name, size in bytes and SHA-256 of every recording."""
    with open(path, newline="") as handle:
        return {row["file"]: (int(row["bytes"]), row["sha256"]) for row in csv.DictReader(handle)}


def check_against_manifest(data_dir, manifest_path):
    """Compare the audio below `data_dir` with the manifest. Returns a list of problems (empty if none)."""
    manifest = read_manifest(manifest_path)
    found = {os.path.basename(p): p for p in find_wav_files(data_dir)}
    problems = [f"missing: {name}" for name in sorted(set(manifest) - set(found))]
    problems += [f"not in the manifest: {name}" for name in sorted(set(found) - set(manifest))]
    for name in sorted(set(manifest) & set(found)):
        size, digest = manifest[name]
        if os.path.getsize(found[name]) != size:
            problems.append(f"size differs: {name}")
        elif sha256_of(found[name]) != digest:
            problems.append(f"content differs: {name}")
    return problems
