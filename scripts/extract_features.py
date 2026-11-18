"""Cut every recording into 10 s clips and compute the frame-level features and the tempo.

    python scripts/extract_features.py --data-dir data/audio

Writes to the cache folder: clips.csv (one row per clip), frames_none.npy (clips, 431, 54), tempo_none.npy,
and for every augmentation arm frames_<arm>.npy, tempo_<arm>.npy, parent_<arm>.npy (index of the original
clip) and kind_<arm>.npy (name of the transform). Writes to the results folder: recordings.csv and clips.csv.
The work is done in batches of recordings that are saved one by one, so an interrupted run continues
where it stopped.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import soundfile
from joblib import Parallel, delayed

from _common import add_common_arguments, log

from humrec import audio, config, data, features


def process_recording(path, file_name, arms, keep_waves=()):
    """All clips of one recording with their features, for the original audio and every arm.

    For the arms in `keep_waves` the changed clips themselves are returned as well (16-bit samples),
    for the models that take the waveform.
    """
    info = soundfile.info(path)
    clips = audio.split_into_clips(audio.load_audio(path))
    out = {"file": file_name, "duration": info.frames / info.samplerate, "sample_rate": info.samplerate,
           "channels": info.channels, "start": [c[0] for c in clips], "real_seconds": [c[2] for c in clips],
           "frames": [], "tempo": [], "augmented": {arm: [] for arm in arms}, "waves": {arm: [] for arm in keep_waves}}
    for index, (_, clip, _) in enumerate(clips):
        out["frames"].append(features.frame_features(clip))
        out["tempo"].append(features.tempo(clip))
        for arm in arms:
            rng = audio.clip_rng(file_name, index, arm)
            for kind, changed in audio.augmented_versions(clip, arm, rng):
                out["augmented"][arm].append((index, kind, features.frame_features(changed), features.tempo(changed)))
                if arm in keep_waves:
                    out["waves"][arm].append(np.clip(np.round(changed * 32767.0), -32768, 32767).astype(np.int16))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="folder with the WAV files")
    parser.add_argument("--arms", nargs="*", default=list(config.AUGMENTATION_ARMS) + list(config.COMBINED_ARMS))
    parser.add_argument("--per-song", type=int, default=None, help="use only the first N recordings of every song")
    parser.add_argument("--batch", type=int, default=80, help="recordings per saved batch")
    parser.add_argument("--keep-waves", nargs="*", default=[],
                        help="arms whose changed clips are also saved as waves_<arm>.npy (for scripts/finetune.py)")
    add_common_arguments(parser)
    args = parser.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    parts_dir = os.path.join(args.cache_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)

    recordings = data.list_recordings(args.data_dir)
    if args.per_song:
        recordings = recordings.groupby("song", sort=False).head(args.per_song).reset_index(drop=True)
    recordings["fold"] = data.assign_folds(recordings)
    log(f"{len(recordings)} recordings, {recordings['participant'].nunique()} participants, "
        f"{args.workers} worker processes, arms: {args.arms}")

    # saved batches are reused only by a run with the same arguments
    settings = {"batch": args.batch, "per_song": args.per_song, "arms": sorted(args.arms),
                "keep_waves": sorted(args.keep_waves), "recordings": len(recordings)}
    settings_path = os.path.join(parts_dir, "settings.json")
    if os.path.exists(settings_path):
        with open(settings_path) as handle:
            if json.load(handle) != settings:
                raise SystemExit(f"{parts_dir} holds batches of a run with other arguments: remove the folder first")
    else:
        with open(settings_path, "w") as handle:
            json.dump(settings, handle)

    features.warm_up()
    batches = [recordings.iloc[i:i + args.batch] for i in range(0, len(recordings), args.batch)]
    for number, batch in enumerate(batches):
        part = os.path.join(parts_dir, f"part_{number:03d}.npy")
        if os.path.exists(part):
            continue
        done = Parallel(n_jobs=args.workers)(
            delayed(process_recording)(row.path, row.file, args.arms, args.keep_waves) for row in batch.itertuples())
        np.save(part + ".tmp.npy", np.array(done, dtype=object), allow_pickle=True)
        os.replace(part + ".tmp.npy", part)
        log(f"batch {number + 1} of {len(batches)} saved")

    by_file = {}
    for number in range(len(batches)):
        for item in np.load(os.path.join(parts_dir, f"part_{number:03d}.npy"), allow_pickle=True):
            by_file[item["file"]] = item

    clip_rows, frames, tempos = [], [], []
    extra = {arm: {"frames": [], "tempo": [], "parent": [], "kind": [], "waves": []} for arm in args.arms}
    stats = []
    for recording, row in enumerate(recordings.itertuples()):
        item = by_file[row.file]
        first_clip = len(clip_rows)
        for index in range(len(item["frames"])):
            clip_rows.append({"clip": len(clip_rows), "recording": recording, "file": row.file,
                              "participant": row.participant, "kind": row.kind, "take": row.take,
                              "song": row.song, "label": row.label, "fold": row.fold,
                              "start_seconds": round(item["start"][index], 3),
                              "real_seconds": round(item["real_seconds"][index], 3)})
            frames.append(item["frames"][index])
            tempos.append(item["tempo"][index])
        for arm in args.arms:
            for index, kind, changed_frames, changed_tempo in item["augmented"][arm]:
                extra[arm]["frames"].append(changed_frames)
                extra[arm]["tempo"].append(changed_tempo)
                extra[arm]["parent"].append(first_clip + index)
                extra[arm]["kind"].append(kind)
            extra[arm]["waves"] += item.get("waves", {}).get(arm, [])
        stats.append({"recording": recording, "file": row.file, "participant": row.participant, "kind": row.kind,
                      "take": row.take, "song": row.song, "label": row.label, "fold": row.fold,
                      "duration_seconds": round(item["duration"], 3), "sample_rate": item["sample_rate"],
                      "channels": item["channels"], "clips": len(item["frames"])})

    clips = pd.DataFrame(clip_rows)
    np.save(os.path.join(args.cache_dir, "frames_none.npy"), np.stack(frames).astype(np.float32))
    np.save(os.path.join(args.cache_dir, "tempo_none.npy"), np.asarray(tempos, dtype=np.float32))
    for arm in args.arms:
        np.save(os.path.join(args.cache_dir, f"frames_{arm}.npy"), np.stack(extra[arm]["frames"]).astype(np.float32))
        np.save(os.path.join(args.cache_dir, f"tempo_{arm}.npy"), np.asarray(extra[arm]["tempo"], dtype=np.float32))
        np.save(os.path.join(args.cache_dir, f"parent_{arm}.npy"), np.asarray(extra[arm]["parent"], dtype=np.int64))
        np.save(os.path.join(args.cache_dir, f"kind_{arm}.npy"), np.asarray(extra[arm]["kind"]))
        if arm in args.keep_waves:
            waves = np.stack(extra[arm]["waves"])
            assert len(waves) == len(extra[arm]["kind"]), (len(waves), len(extra[arm]["kind"]))
            np.save(os.path.join(args.cache_dir, f"waves_{arm}.npy"), waves)
        counts = pd.Series(extra[arm]["kind"]).value_counts().to_dict()
        log(f"arm {arm}: {len(extra[arm]['kind'])} added clips {counts}")
    clips.to_csv(os.path.join(args.cache_dir, "clips.csv"), index=False)
    clips.to_csv(os.path.join(args.results_dir, "clips.csv"), index=False)
    pd.DataFrame(stats).to_csv(os.path.join(args.results_dir, "recordings.csv"), index=False)
    per_fold = clips.groupby("fold").size().to_dict()
    log(f"{len(clips)} clips from {len(recordings)} recordings, clips per fold: {per_fold}")
    dropped = int((pd.DataFrame(stats)["clips"] == 0).sum())
    log(f"recordings without a clip: {dropped}")


if __name__ == "__main__":
    main()
