"""Recognise the song by matching pitch contours with dynamic time warping (query by humming).

    python scripts/run_melody.py --data-dir data/audio

The pitch contour of a clip (from f0.npy, written by extract_pitch.py) is compared with the pitch contour
of every whole training recording: the clip may match any part of the recording, in another key and at
up to twice or half the tempo. The nearest training recordings vote for their song. The number of
voters and the temperature of the vote weights are chosen per fold by the clip accuracy on the
validation fold. Nothing is trained and nothing is random, so there is one run per fold and no seed.

The pitch tracks of the whole recordings are computed once and kept in the cache folder as
f0_recordings.npz (one array per file name). Only this step reads the audio.
The predictions are saved to results/predictions/knn__contour_dtw__none.npz.
"""

import argparse
import os
import time

import librosa
import numpy as np
from joblib import Parallel, delayed

from _common import add_common_arguments, log

from humrec import audio, config, data, evaluation, features, melody

NAME = "knn__contour_dtw__none"


def recording_pitch(path):
    """Pitch track of one whole recording, with the silence at both ends cut as it is for the clips."""
    samples, _ = librosa.effects.trim(audio.load_audio(path), top_db=config.TRIM_TOP_DB)
    return features.pitch_track(samples, n_frames=1 + len(samples) // config.HOP_LENGTH)


def recording_tracks(cache_dir, data_dir, files, workers):
    """Pitch track of every recording by file name, read from f0_recordings.npz or computed and stored there."""
    path = os.path.join(cache_dir, "f0_recordings.npz")
    if not os.path.exists(path):
        if data_dir is None:
            raise SystemExit("f0_recordings.npz is not in the cache yet, pass --data-dir to compute it")
        paths = {os.path.basename(p): p for p in data.find_wav_files(data_dir)}
        features.warm_up()
        tracks = Parallel(n_jobs=workers)(delayed(recording_pitch)(paths[f]) for f in files)
        np.savez_compressed(path + ".tmp.npz", **dict(zip(files, tracks)))
        os.replace(path + ".tmp.npz", path)
        log(f"f0_recordings.npz written, {len(files)} recordings")
    with np.load(path) as stored:
        missing = [f for f in files if f not in stored.files]
        if missing:
            raise RuntimeError(f"{path} lacks {len(missing)} recordings of clips.csv, delete it to compute it again")
        return {f: stored[f] for f in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None,
                        help="folder with the WAV files (not needed once f0_recordings.npz is in the cache)")
    parser.add_argument("--neighbours", nargs="*", type=int, default=list(melody.K_GRID),
                        help="numbers of voting recordings to compare on the validation fold")
    parser.add_argument("--temperatures", nargs="*", type=float, default=list(melody.TEMPERATURE_GRID),
                        help="temperatures of the vote weights (semitones) to compare on the validation fold")
    parser.add_argument("--transpositions", nargs="*", type=float, default=list(melody.TRANSPOSITIONS),
                        help="all semitone shifts of the query tried in the key search (12 and -12 can be added "
                             "against octave errors of the pitch tracker)")
    add_common_arguments(parser)
    args = parser.parse_args()
    clips = evaluation.load_clips(os.path.join(args.cache_dir, "clips.csv"))
    f0 = np.load(os.path.join(args.cache_dir, "f0.npy"))
    assert len(f0) == len(clips), (len(f0), len(clips))
    recordings = clips.drop_duplicates("recording").reset_index(drop=True)
    labels = clips["label"].to_numpy()
    log(f"{len(clips)} clips of {len(recordings)} recordings, {args.workers} workers")

    tracks = recording_tracks(args.cache_dir, args.data_dir, list(recordings["file"]), args.workers)
    queries = [melody.contour(row) for row in f0]
    references = [melody.contour(tracks[f]) for f in recordings["file"]]
    empty = {"queries": sum(len(c) == 0 for c in queries), "references": sum(len(c) == 0 for c in references)}
    log(f"contours: clips {np.mean([len(c) for c in queries]):.0f} values on average, recordings "
        f"{np.mean([len(c) for c in references]):.0f}, empty: {empty}")

    # One matrix for all folds. A distance depends only on its clip and its recording, so the same
    # numbers serve every fold, and each fold reads only the columns of its training recordings.
    started = time.time()
    distances = melody.distance_matrix(queries, references, args.transpositions, n_jobs=args.workers)
    log(f"{distances.size} distances in {time.time() - started:.1f} s, "
        f"share without an alignment {np.mean(~np.isfinite(distances)):.4f}")

    proba = np.full((1, len(clips), config.N_CLASSES), np.nan, dtype=np.float32)
    val_accuracy = np.full((1, config.N_FOLDS), np.nan, dtype=np.float32)
    train_accuracy = np.full((1, config.N_FOLDS), np.nan, dtype=np.float32)   # nothing is fitted to the training set
    runs = []
    for fold in range(config.N_FOLDS):
        train, _, _ = data.fold_roles(recordings["fold"], fold)   # the references: training recordings only
        _, val, test = data.fold_roles(clips["fold"], fold)
        reference_labels = recordings["label"].to_numpy()[train]
        k, temperature, table = melody.choose_settings(distances[val][:, train], labels[val], reference_labels,
                                                       args.neighbours, args.temperatures)
        proba[0, test] = melody.predict(distances[test][:, train], reference_labels, k, temperature)
        val_accuracy[0, fold] = table.max()
        runs.append({"fold": fold, "k": k, "temperature": temperature, "n_references": int(train.sum()),
                     "n_validation_clips": int(val.sum()), "n_test_clips": int(test.sum()),
                     "val_accuracy_grid": np.round(table, 4).tolist()})
    if np.isnan(proba).any():
        raise RuntimeError("some clips have no prediction")

    info = {"experiment": NAME, "model": "knn", "input": "contour_dtw", "arm": "none",
            "contour": {"values_per_second": melody.FRAME_RATE / melody.DOWNSAMPLE,
                        "median_frames": melody.MEDIAN_FRAMES, "downsample": melody.DOWNSAMPLE,
                        "max_semitones": melody.MAX_SEMITONES, "min_voiced_seconds": melody.MIN_VOICED_SECONDS,
                        "unvoiced_frames": "removed"},
            "dtw": {"steps": "(1,1), (1,2), (2,1)", "cost": "mean absolute semitone difference per query value",
                    "reference": "whole training recordings, free start and end"},
            "transpositions": list(args.transpositions), "k_grid": list(args.neighbours),
            "temperature_grid": list(args.temperatures), "empty_contours": empty, "runs": runs}
    evaluation.save_predictions(os.path.join(args.results_dir, "predictions"), NAME, proba, val_accuracy,
                                train_accuracy, info)
    correct = proba[0].argmax(axis=1) == labels
    per_fold = [float(correct[clips["fold"].to_numpy() == fold].mean()) for fold in range(config.N_FOLDS)]
    log(f"{NAME}: test clip accuracy {correct.mean():.4f} (per fold {np.round(per_fold, 4).tolist()}), "
        f"validation {val_accuracy.mean():.4f} (per fold {[round(float(v), 4) for v in val_accuracy[0]]})")
    log(f"{NAME}: chosen per fold (k, temperature) {[(r['k'], r['temperature']) for r in runs]}")


if __name__ == "__main__":
    main()
