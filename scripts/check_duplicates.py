"""Look for the same performance under two participant numbers, with the saved pitch tracks.

    python scripts/check_duplicates.py

Participant-disjoint folds protect against a leak only if one participant number is one person and no
recording was handed in twice. Two files with the same bytes are excluded by data/manifest.csv. This
script looks for copies that differ in their bytes (another format, another loudness, a shift of the
pitch as a whole): for every pair of clips it takes the frames in which both are voiced and computes the
standard deviation of the difference of the two pitch tracks in semitones. The same audio gives a value
near 0 even after a pitch shift, two different performances give a few semitones. Clips of two copies
are cut at the same places, so they are compared frame by frame.

Reads f0.npy and clips.csv from the cache folder (written by scripts/extract_pitch.py) and writes
results/duplicate_check.json with the smallest values for pairs of clips of different participants.
Needs no GPU and takes a second.
"""

import argparse
import json
import os

import numpy as np

from _common import add_common_arguments, log

from humrec import evaluation, features


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-common", type=int, default=150, help="frames that both clips of a pair must have voiced")
    add_common_arguments(parser)
    args = parser.parse_args()
    clips = evaluation.load_clips(os.path.join(args.cache_dir, "clips.csv"))
    f0 = np.load(os.path.join(args.cache_dir, "f0.npy"))
    spread, common = features.pitch_difference_spread(f0, args.min_common)
    participant = clips["participant"].to_numpy()
    recording = clips["recording"].to_numpy()
    upper = np.triu(np.ones(spread.shape, dtype=bool), 1)
    groups = {"different_participants": participant[:, None] != participant[None, :],
              "same_participant_other_recording": (participant[:, None] == participant[None, :])
              & (recording[:, None] != recording[None, :])}
    out = {"min_common_frames": args.min_common, "clips": int(len(clips))}
    for name, mask in groups.items():
        values = np.where(mask & upper, spread, np.inf)
        finite = np.sort(values[np.isfinite(values)])
        order = np.argsort(values, axis=None)[:5]
        pairs = []
        for a, b in zip(*np.unravel_index(order, values.shape)):
            pairs.append({"file_a": clips["file"][a], "start_a": float(clips["start_seconds"][a]),
                          "file_b": clips["file"][b], "start_b": float(clips["start_seconds"][b]),
                          "spread_semitones": float(values[a, b]), "common_frames": int(common[a, b])})
        out[name] = {"pairs_compared": int(len(finite)), "smallest_spread_semitones": float(finite[0]),
                     "median_spread_semitones": float(np.median(finite)),
                     "pairs_below_a_tenth_of_a_semitone": int((finite < 0.1).sum()), "closest_pairs": pairs}
        log(f"{name}: {len(finite)} pairs, smallest spread {finite[0]:.3f} semitones, median {np.median(finite):.2f}, "
            f"pairs below 0.1 semitones: {int((finite < 0.1).sum())}")
    with open(os.path.join(args.results_dir, "duplicate_check.json"), "w") as handle:
        json.dump(out, handle, indent=1)


if __name__ == "__main__":
    main()
