"""Tables and figures built from the saved predictions. Used by scripts/make_report.py and the notebook."""

import glob
import json
import os

import numpy as np
import pandas as pd

from . import config, evaluation, experiments

MODEL_LABELS = {"random_forest": "Random forest", "svm": "SVM (RBF kernel)", "xgboost": "XGBoost",
                "lightgbm": "LightGBM", "cnn1d": "1D CNN", "cnn2d": "2D CNN", "bilstm": "Bidirectional LSTM",
                "bilstm_pool": "Bidirectional LSTM, mean over time",
                "keynet": "Key-invariant 2D CNN", "keynet12": "Key-invariant 2D CNN, full circle",
                "mlp": "MLP on frozen encoder", "finetune": "Fine-tuned encoder", "knn": "Nearest neighbours",
                "ensemble": "Ensemble"}
INPUT_LABELS = {"aggregate": "mean and std over time (109 numbers)",
                "temporal": "mean and std in N parts of the clip, selected",
                "temporal_tuned": "mean and std in N parts of the clip, selected, tuned XGBoost settings",
                "contour": "pitch contour", "contour_dtw": "pitch contour, DTW to the training recordings",
                "contour_wide": "pitch contour, tracker range up to 4186 Hz",
                "melody_full": "chroma, chroma difference, onset, tempo (26 channels)",
                "melody": "chroma, onset, tempo (14 channels)", "frames": "all 54 frame-level features",
                "ast": "AST, classification token of the last layer", "ast_layers": "AST, layer chosen on validation",
                "wav2vec2": "wav2vec 2.0 base", "hubert": "HuBERT base", "wavlm": "WavLM base plus",
                "whisper": "Whisper small encoder", "mert": "MERT 95M", "mert330": "MERT 330M",
                "features": "three models on the frame-level features", "all": "six models, all inputs",
                "best": "best model of every family by validation accuracy",
                "top2": "best models of the two families with the highest validation accuracy"}
SHORT_INPUT_LABELS = {"aggregate": "aggregate statistics", "temporal": "statistics of N parts",
                      "temporal_tuned": "statistics of N parts, tuned", "contour_dtw": "pitch contour with DTW",
                      "melody_full": "chroma, its difference, onset, tempo", "melody": "chroma, onset, tempo",
                      "frames": "54 frame-level features", "ast": "AST (last token)", "ast_layers": "AST (best layer)",
                      "features": "3 feature models", "all": "6 models", "best": "best of every family",
                      "top2": "best of the two strongest families"}
ARM_LABELS = {"none": "", "conservative": "conservative augmentation", "strong": "strong augmentation",
              "conservative_noise": "conservative augmentation, noise only",
              "conservative_pitch": "conservative augmentation, pitch shift only",
              "conservative_repeat": "no augmentation, clips repeated as often as in the conservative arm",
              "heavy": "combined augmentation, four copies per clip",
              "strong_noise": "strong augmentation, noise only",
              "strong_pitch": "strong augmentation, pitch shift only",
              "strong_stretch": "strong augmentation, time stretch only"}
VARIANT_LABELS = {"two_stage": "two stages: hum or whistle first", "kind_norm": "standardised per type",
                  "mixup": "mixup and label smoothing", "pruned": "least recognisable training recordings removed",
                  "stretch": "random tempo changes in training", "transpose": "random key changes in training",
                  "transpose_long": "random key changes in training, up to 300 epochs"}
ENCODER_INPUTS = ("ast", "ast_layers", "wav2vec2", "hubert", "wavlm", "whisper", "mert", "mert330")


def split_name(name):
    """(model, input, variant, arm) keys of an experiment name."""
    parts = name.split("__")
    if parts[0] == "ensemble":
        return "ensemble", parts[1], "", "none"
    model, inputs, arm = parts[:3]
    variant = ""
    for key in VARIANT_LABELS:
        if inputs.endswith("_" + key):
            inputs, variant = inputs[:-len(key) - 1], key
    return model, inputs, variant, arm


def family(name):
    """Family of an experiment, used to pick the members of the `best` ensemble."""
    model, inputs, _, _ = split_name(name)
    if model == "ensemble":
        return "ensemble"
    if model == "finetune":
        return "fine-tuned encoder"
    if model == "knn":
        return "melody matching"
    if inputs in ENCODER_INPUTS:
        return "frozen encoder"
    if inputs in ("contour", "contour_wide"):
        return "pitch contour"
    return "hand-made features"


# Pairs (a, b) for which the difference accuracy(a) - accuracy(b) is reported, next to the pairs that
# variant_pairs() derives from the names (a variant or an augmentation arm against the plain experiment).
COMPARISONS = (
    [(f"{m}__aggregate__{arm}", f"{m}__aggregate__none") for m in ("random_forest", "svm", "xgboost", "lightgbm")
     for arm in ("conservative", "strong")]
    + [(f"cnn1d__melody_full__{arm}", "cnn1d__melody_full__none") for arm in ("conservative", "strong")]
    + [("xgboost__temporal__none", "xgboost__aggregate__none"),
       ("xgboost__temporal_tuned__none", "xgboost__temporal__none"),
       ("cnn1d__melody_full__none", "cnn1d__melody__none"),
       ("cnn1d__melody__none", "xgboost__temporal__none"),
       ("cnn1d__melody_full__none", "cnn1d__frames__none"),
       ("cnn2d__melody_full__none", "cnn1d__melody_full__none"),
       ("cnn1d__frames__none", "xgboost__aggregate__none"),
       ("bilstm__frames__none", "xgboost__aggregate__none"),
       ("xgboost__contour__none", "xgboost__aggregate__none"),
       ("mlp__ast__none", "xgboost__aggregate__none"),
       ("ensemble__features", "xgboost__aggregate__none"),
       ("ensemble__all", "xgboost__aggregate__none"),
       ("ensemble__all", "mlp__ast__none"),
       ("mlp__ast_layers__none", "mlp__ast__none"),
       ("knn__contour_dtw__none", "xgboost__contour__none"),
       ("ensemble__best", "ensemble__all")]
    + [(f"mlp__{e}__none", "mlp__ast__none") for e in ("wav2vec2", "hubert", "wavlm", "whisper", "mert", "mert330")]
    + [("finetune__mert__none", "mlp__mert__none"),
       ("cnn1d__contour_wide__none", "cnn1d__contour__none"),
       ("knn__contour_dtw__none", "cnn1d__contour__none")]
    + [(f"bilstm_pool__{i}__none", f"bilstm__{i}__none") for i in ("frames", "contour")]
    + [(f"keynet12__{i}__{arm}", f"keynet__{i}__{arm}")
       for i, arm in (("melody", "none"), ("melody_full", "none"), ("melody", "strong"))]
    + [(f"cnn1d__{i}_transpose_long__{arm}", f"cnn1d__{i}_transpose__{arm}")
       for i, arm in (("melody", "none"), ("melody_full", "none"), ("melody", "strong"))]
    + [(f"keynet__{i}__none", f"cnn1d__{i}_transpose__none") for i in ("melody", "melody_full")]
)


def label(name):
    """Readable (model, input, variant) of an experiment name. The variant is empty for the plain experiment."""
    model, inputs, variant, arm = split_name(name)
    note = ", ".join(text for text in (VARIANT_LABELS.get(variant, ""), ARM_LABELS.get(arm, "")) if text)
    return MODEL_LABELS[model], INPUT_LABELS[inputs], note


def best_members(results_dir, top=None):
    """The experiment with the highest mean validation accuracy in every family (checks and ensembles excluded).

    With `top` only the families with the `top` highest validation accuracies are returned.
    """
    best = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "predictions", "*.npz"))):
        name = os.path.basename(path)[:-4]
        if name.count("__") != 2 or name.startswith("ensemble"):
            continue
        score = float(np.nanmean(evaluation.load_predictions(path)["val_accuracy"]))
        group = family(name)
        if group not in best or score > best[group][1]:
            best[group] = (name, score)
    groups = sorted(best, key=lambda g: -best[g][1])[:top] if top else sorted(best)
    return {group: best[group][0] for group in groups}


def build_ensembles(results_dir):
    """Average the saved probabilities of the members of every ensemble and save them like an experiment."""
    folder = os.path.join(results_dir, "predictions")
    written = []
    ensembles = dict(experiments.ENSEMBLES)
    chosen = best_members(results_dir)
    if len(chosen) > 1:
        ensembles["ensemble__best"] = tuple(chosen.values())
        ensembles["ensemble__top2"] = tuple(best_members(results_dir, top=2).values())
    for name, members in ensembles.items():
        paths = [os.path.join(folder, m + ".npz") for m in members]
        if not all(os.path.exists(p) for p in paths):
            continue
        loaded = [evaluation.load_predictions(p) for p in paths]
        n_seeds = max(m["proba"].shape[0] for m in loaded)
        # a member without random seeds has one set of predictions, which is used with every seed of the others
        proba = np.mean([np.broadcast_to(m["proba"], (n_seeds,) + m["proba"].shape[1:]) if m["proba"].shape[0] == 1
                         else m["proba"] for m in loaded], axis=0)
        val = np.full_like(loaded[0]["val_accuracy"], np.nan)
        info = {"experiment": name, "members": list(members), "rule": "mean of the class probabilities"}
        evaluation.save_predictions(folder, name, proba, val, val, info)
        written.append(name)
    return written


def load_all(results_dir):
    """Clip table and a dict name -> predictions for every file in results/predictions."""
    clips = evaluation.load_clips(os.path.join(results_dir, "clips.csv"))
    predictions = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "predictions", "*.npz"))):
        predictions[os.path.basename(path)[:-4]] = evaluation.load_predictions(path)
    return clips, predictions


def correct_share(proba, clips, level="clip"):
    """Per item, the share of seeds whose prediction is right, and the participant of the item."""
    if level == "clip":
        right = proba.argmax(axis=2) == clips["label"].to_numpy()[None, :]
        return right.mean(axis=0), clips["participant"].to_numpy()
    index = pd.factorize(clips["recording"], sort=True)[0]
    first = clips.drop_duplicates("recording").sort_values("recording")
    right = evaluation.recording_proba(proba, index).argmax(axis=2) == first["label"].to_numpy()[None, :]
    return right.mean(axis=0), first["participant"].to_numpy()


def _mean_or_nan(values):
    """Mean of the finite values, NaN if there is none (ensembles have no validation or training accuracy)."""
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def model_table(clips, predictions):
    """One row per experiment with accuracy at clip and recording level (fractions, not percent)."""
    rows = []
    for name, item in predictions.items():
        if name.count("__") == 3:
            continue  # protocol checks have their own table
        model, inputs, note = label(name)
        summary = evaluation.summarise(item["proba"], clips)
        rows.append({
            "experiment": name, "family": family(name), "model": model, "input": inputs, "variant": note,
            "seeds": int(item["proba"].shape[0]),
            "clip_accuracy": summary["clip"]["mean"], "clip_std": summary["clip"]["std"],
            "clip_ci_low": summary["clip"]["ci95"][0], "clip_ci_high": summary["clip"]["ci95"][1],
            "clip_fold_min": min(summary["clip"]["per_fold"]), "clip_fold_max": max(summary["clip"]["per_fold"]),
            "recording_accuracy": summary["recording"]["mean"], "recording_std": summary["recording"]["std"],
            "recording_ci_low": summary["recording"]["ci95"][0], "recording_ci_high": summary["recording"]["ci95"][1],
            "validation_accuracy": _mean_or_nan(item["val_accuracy"]),
            "train_accuracy": _mean_or_nan(item["train_accuracy"]),
        })
    table = pd.DataFrame(rows)
    return table.sort_values("clip_accuracy", ascending=False).reset_index(drop=True)


def final_model(table):
    """The single experiment with the highest mean validation accuracy (ensembles have no validation accuracy)."""
    singles = table[~table["experiment"].str.startswith("ensemble") & table["validation_accuracy"].notna()]
    return str(singles.sort_values("validation_accuracy", ascending=False).iloc[0]["experiment"])


CHECK_LABELS = {"shuffled_labels": "training labels shuffled",
                "recording_folds": "recordings split at random (participants on both sides)",
                "clip_folds": "clips split at random (recordings on both sides)"}


def checks_table(clips, predictions):
    """Accuracy of the protocol checks next to the accuracy of the same model under the real protocol."""
    rows = []
    for name, item in predictions.items():
        if name.count("__") != 3:
            continue
        base, check = name.rsplit("__", 1)
        summary = evaluation.summarise(item["proba"], clips)
        reference = evaluation.summarise(predictions[base]["proba"], clips) if base in predictions else None
        rows.append({
            "experiment": name, "model": _short(base), "check": CHECK_LABELS[check],
            "seeds": int(item["proba"].shape[0]),
            "clip_accuracy": summary["clip"]["mean"], "clip_std": summary["clip"]["std"],
            "clip_ci_low": summary["clip"]["ci95"][0], "clip_ci_high": summary["clip"]["ci95"][1],
            "recording_accuracy": summary["recording"]["mean"], "recording_std": summary["recording"]["std"],
            "clip_accuracy_participant_folds": reference["clip"]["mean"] if reference else np.nan,
            "recording_accuracy_participant_folds": reference["recording"]["mean"] if reference else np.nan,
        })
    return pd.DataFrame(rows)


def variant_pairs(predictions):
    """Pairs (changed experiment, plain experiment) for every saved experiment that has a variant or an
    augmentation arm: the variant against the same model without it, the arm against no augmentation."""
    pairs = []
    for name in sorted(predictions):
        if name.count("__") != 2 or name.startswith("ensemble"):
            continue
        model, inputs, variant, arm = split_name(name)
        full_input = name.split("__")[1]
        if variant:
            pairs.append((name, f"{model}__{inputs}__{arm}"))
        if arm != "none":
            pairs.append((name, f"{model}__{full_input}__none"))
    return [pair for pair in dict.fromkeys(pairs) if pair not in COMPARISONS]


def comparison_table(clips, predictions, level="clip"):
    """Accuracy differences for the pairs in COMPARISONS with participant-bootstrap intervals."""
    rows = []
    for a, b in list(COMPARISONS) + variant_pairs(predictions):
        if a not in predictions or b not in predictions:
            continue
        share_a, groups = correct_share(predictions[a]["proba"], clips, level)
        share_b, _ = correct_share(predictions[b]["proba"], clips, level)
        diff, lo, hi = evaluation.bootstrap_difference(share_a, share_b, groups)
        rows.append({"a": a, "b": b, "difference": diff, "ci_low": lo, "ci_high": hi,
                     "interval_excludes_zero": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def repeat_table(clips, predictions, results_dir):
    """Experiments that were trained a second time on another machine (results/repeat/predictions)."""
    rows = []
    labels = clips["label"].to_numpy()[None, :]
    for path in sorted(glob.glob(os.path.join(results_dir, "repeat", "predictions", "*.npz"))):
        name = os.path.basename(path)[:-4]
        if name not in predictions:
            continue
        first = predictions[name]["proba"].astype(np.float64)
        second = evaluation.load_predictions(path)["proba"].astype(np.float64)
        if first.shape != second.shape:
            continue
        rows.append({
            "experiment": name,
            "clip_accuracy": float((first.argmax(axis=2) == labels).mean()),
            "clip_accuracy_repeated": float((second.argmax(axis=2) == labels).mean()),
            "same_prediction_share": float((first.argmax(axis=2) == second.argmax(axis=2)).mean()),
            "max_probability_difference": float(np.abs(first - second).max()),
        })
    return pd.DataFrame(rows, columns=["experiment", "clip_accuracy", "clip_accuracy_repeated", "same_prediction_share",
                                       "max_probability_difference"])


def _best_of_families(predictions, fold=None):
    """Family -> (experiment, validation accuracy) of the best experiment of the family.

    With `fold` the validation accuracy of that split alone is used, otherwise the mean of all splits.
    """
    best = {}
    for name in sorted(predictions):
        if name.count("__") != 2 or name.startswith("ensemble"):
            continue
        scores = np.nanmean(predictions[name]["val_accuracy"], axis=0)
        score = float(np.nanmean(scores) if fold is None else scores[fold])
        group = family(name)
        if group not in best or score > best[group][1]:
            best[group] = (name, score)
    return best


def selection_check(clips, predictions):
    """The choices between experiments, made once for all folds and made strictly per test fold.

    The final model and the members of the two ensembles that are defined by validation accuracy are
    chosen with the mean validation accuracy of the five splits. Every fold is the validation fold of one
    split, so the labels of every fold enter these choices. Here the same choices are made again for
    every test fold with the validation accuracy of its own split alone (a fold that the tested model
    never saw and that is not the test fold), and the chosen test predictions are put together.
    """
    folds = clips["fold"].to_numpy()
    n_seeds = max(item["proba"].shape[0] for item in predictions.values())
    def ranked(best):
        return [name for name, _ in sorted(best.values(), key=lambda pair: -pair[1])]

    rules = {"final model": lambda best: ranked(best)[:1], "two strongest families": lambda best: ranked(best)[:2],
             "best model of every family": ranked}
    rows = []
    for title, pick in rules.items():
        result = {"choice": title}
        for column, per_fold in (("all_splits", False), ("own_split", True)):
            proba = np.zeros((n_seeds,) + next(iter(predictions.values()))["proba"].shape[1:], dtype=np.float64)
            chosen = []
            for k in sorted(set(folds.tolist())):
                members = pick(_best_of_families(predictions, k if per_fold else None))
                chosen.append(" + ".join(sorted(members)))
                rows_of_fold = folds == k
                stack = [np.broadcast_to(predictions[m]["proba"], proba.shape)[:, rows_of_fold].astype(np.float64)
                         for m in members]
                proba[:, rows_of_fold] = np.mean(stack, axis=0)
            summary = evaluation.summarise(proba, clips)
            result[f"clip_accuracy_{column}"] = summary["clip"]["mean"]
            result[f"recording_accuracy_{column}"] = summary["recording"]["mean"]
            if per_fold:
                result["chosen_per_fold"] = " | ".join(chosen)
        rows.append(result)
    return pd.DataFrame(rows)


def padded_table(clips, predictions, names):
    """Clip accuracy on the full 10 s clips and on the padded last clip of every recording."""
    rows = []
    padded = clips["real_seconds"].to_numpy() < config.CLIP_SECONDS - 1e-6
    for name in names:
        if name not in predictions:
            continue
        share, _ = correct_share(predictions[name]["proba"], clips)
        rows.append({"experiment": name, "full": float(share[~padded].mean()), "padded": float(share[padded].mean())})
    return pd.DataFrame(rows, columns=["experiment", "full", "padded"])


def kind_table(clips, predictions, names):
    """Clip accuracy on hummed and on whistled clips."""
    rows = []
    kinds = clips["kind"].to_numpy()
    for name in names:
        if name not in predictions:
            continue
        share, _ = correct_share(predictions[name]["proba"], clips)
        rows.append({"experiment": name, "hum": float(share[kinds == "hum"].mean()),
                     "whistle": float(share[kinds == "whistle"].mean())})
    return pd.DataFrame(rows)


def confusion(proba, clips):
    """Confusion matrix of the clips summed over seeds, rows are the true song. Returns counts."""
    labels = clips["label"].to_numpy()
    matrix = np.zeros((config.N_CLASSES, config.N_CLASSES), dtype=np.int64)
    for seed in range(proba.shape[0]):
        np.add.at(matrix, (labels, proba[seed].argmax(axis=1)), 1)
    return matrix


def data_summary(results_dir):
    """Numbers that describe the data set, from results/recordings.csv and results/clips.csv."""
    recordings = pd.read_csv(os.path.join(results_dir, "recordings.csv"))
    clips = pd.read_csv(os.path.join(results_dir, "clips.csv"))
    duration = recordings["duration_seconds"]
    per_participant = recordings.groupby("participant").size()
    return {
        "recordings": int(len(recordings)), "participants": int(recordings["participant"].nunique()),
        "songs": int(recordings["song"].nunique()),
        "recordings_per_song": recordings.groupby("song").size().to_dict(),
        "hum": int((recordings["kind"] == "hum").sum()), "whistle": int((recordings["kind"] == "whistle").sum()),
        "duration_min": float(duration.min()), "duration_median": float(duration.median()),
        "duration_max": float(duration.max()), "hours": float(duration.sum() / 3600),
        "duration_median_by_song": {song: float(value) for song, value
                                    in recordings.groupby("song")["duration_seconds"].median().items()},
        "sample_rates": {int(k): int(v) for k, v in recordings["sample_rate"].value_counts().items()},
        "channels": {int(k): int(v) for k, v in recordings["channels"].value_counts().items()},
        "recordings_per_participant_median": float(per_participant.median()),
        "recordings_per_participant_min": int(per_participant.min()),
        "participant_song_pairs": int(len(recordings.groupby(["participant", "song"]))),
        "participant_song_pairs_recorded_twice": int((recordings.groupby(["participant", "song"]).size() > 1).sum()),
        "participants_with_hum_and_whistle": int((recordings.groupby("participant")["kind"].nunique() == 2).sum()),
        "recordings_per_participant_max": int(per_participant.max()),
        "clips": int(len(clips)), "clips_per_recording_mean": float(len(clips) / len(recordings)),
        "padded_clips": int((clips["real_seconds"] < config.CLIP_SECONDS - 1e-6).sum()),
        "recordings_without_clip": int((recordings["clips"] == 0).sum()),
        "clips_per_fold": {int(k): int(v) for k, v in clips.groupby("fold").size().items()},
        "recordings_per_fold": {int(k): int(v) for k, v in recordings.groupby("fold").size().items()},
        "participants_per_fold": {int(k): int(v)
                                  for k, v in recordings.groupby("fold")["participant"].nunique().items()},
    }


# Figures

SURFACE, INK, INK_SOFT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _axes(width, height):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, length=0, labelsize=9)
    return fig, ax


def _short(name):
    model, inputs, variant, _ = split_name(name)
    model_text = MODEL_LABELS[model].replace(" on frozen encoder", "")
    text = f"{model_text}, {SHORT_INPUT_LABELS.get(inputs, INPUT_LABELS[inputs])}"
    return f"{text}, {VARIANT_LABELS[variant]}" if variant else text


def figure_overview(table, path=None):
    """Clip accuracy of every model trained without augmentation, with the participant-bootstrap interval."""
    rows = table[table["variant"] == ""].sort_values("clip_accuracy")
    fig, ax = _axes(8.2, 0.34 * len(rows) + 1.3)
    y = np.arange(len(rows))
    is_ensemble = rows["experiment"].str.startswith("ensemble").to_numpy()
    colours = np.where(is_ensemble, ORANGE, BLUE)
    ax.hlines(y, 100 * rows["clip_ci_low"], 100 * rows["clip_ci_high"], color=colours, linewidth=2, alpha=0.45)
    ax.scatter(100 * rows["clip_accuracy"], y, s=46, c=colours, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    ax.axvline(100 * config.CHANCE_LEVEL, color=GREY, linewidth=1)
    ax.text(100 * config.CHANCE_LEVEL, len(rows) - 0.35, " chance 12.5%", color=INK_SOFT, fontsize=8.5, va="bottom")
    for position, row in zip(y, rows.itertuples()):
        ax.text(100 * row.clip_ci_high + 0.6, position, f"{100 * row.clip_accuracy:.1f}", color=INK, fontsize=8.5,
                va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([_short(n) for n in rows["experiment"]], color=INK, fontsize=9)
    ax.set_ylim(-0.7, len(rows) + 0.2)
    ax.set_xlim(0, max(60, 100 * rows["clip_ci_high"].max() + 6))
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"accuracy on 10 s clips, % (dot: mean of {int(rows['seeds'].max())} seeds, "
                  "line: 95% interval over participants)", color=INK_SOFT, fontsize=9)
    ax.set_title("Test accuracy by model, no augmentation (ensembles in orange)", color=INK, fontsize=11, loc="left")
    return _finish(fig, path)


def figure_augmentation(table, path=None):
    """Clip accuracy of five models under the three augmentation arms."""
    models = ["random_forest__aggregate", "svm__aggregate", "xgboost__aggregate", "lightgbm__aggregate",
              "cnn1d__melody_full"]
    arms = [("none", BLUE, "o"), ("conservative", ORANGE, "s"), ("strong", AQUA, "D")]
    fig, ax = _axes(8.2, 3.6)
    indexed = table.set_index("experiment")
    for k, (arm, colour, shape) in enumerate(arms):
        xs, means, lows, highs = [], [], [], []
        for position, model in enumerate(models):
            name = f"{model}__{arm}"
            if name not in indexed.index:
                continue
            row = indexed.loc[name]
            xs.append(position + (k - 1) * 0.22)
            means.append(100 * row["clip_accuracy"])
            lows.append(100 * row["clip_ci_low"])
            highs.append(100 * row["clip_ci_high"])
        ax.vlines(xs, lows, highs, color=colour, linewidth=2, alpha=0.45)
        ax.scatter(xs, means, s=46, color=colour, marker=shape, edgecolors=SURFACE, linewidths=1.5, zorder=3,
                   label=arm)
    ax.axhline(100 * config.CHANCE_LEVEL, color=GREY, linewidth=1)
    ax.text(len(models) - 0.55, 100 * config.CHANCE_LEVEL + 0.5, "chance", color=INK_SOFT, fontsize=8.5, ha="right")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(["Random forest", "SVM", "XGBoost", "LightGBM", "1D CNN"], color=INK, fontsize=9)
    ax.set_ylim(0, None)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_ylabel("accuracy on 10 s clips, %", color=INK_SOFT, fontsize=9)
    legend = ax.legend(title="augmentation of the training clips", frameon=False, fontsize=9, ncol=3,
                       loc="lower left", bbox_to_anchor=(0, 1.0), alignment="left")
    legend.get_title().set_color(INK_SOFT)
    legend.get_title().set_fontsize(9)
    ax.set_title(f"Augmentation arms (mean of {int(table['seeds'].max())} seeds, 95% interval over participants)",
                 color=INK, fontsize=11, loc="left", pad=44)
    return _finish(fig, path)


def figure_confusion(matrix, title, path=None):
    """Row-normalised confusion matrix: of the clips of one song, the share assigned to each song."""
    from matplotlib.colors import LinearSegmentedColormap
    share = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = _axes(6.2, 5.4)
    image = ax.imshow(100 * share, cmap=LinearSegmentedColormap.from_list("blue", BLUE_RAMP), vmin=0, vmax=100)
    for i in range(config.N_CLASSES):
        for j in range(config.N_CLASSES):
            value = 100 * share[i, j]
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if value > 45 else INK)
    ax.set_xticks(range(config.N_CLASSES))
    ax.set_yticks(range(config.N_CLASSES))
    ax.set_xticklabels(config.SONGS, rotation=40, ha="right", color=INK, fontsize=9)
    ax.set_yticklabels(config.SONGS, color=INK, fontsize=9)
    ax.set_xlabel("predicted song", color=INK_SOFT, fontsize=9)
    ax.set_ylabel("true song", color=INK_SOFT, fontsize=9)
    ax.spines["bottom"].set_visible(False)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=INK_SOFT, length=0, labelsize=8.5)
    bar.set_label("% of the clips of the true song", color=INK_SOFT, fontsize=9)
    return _finish(fig, path)


def figure_parts(sweep, path=None):
    """Validation accuracy of XGBoost against the number of temporal parts, one thin line per fold."""
    parts = sweep["parts"]
    values = 100 * np.asarray(sweep["val_accuracy"])
    fig, ax = _axes(6.4, 3.3)
    for row in values:
        ax.plot(range(len(parts)), row, color=GREY, linewidth=1, alpha=0.7)
    ax.plot(range(len(parts)), values.mean(axis=0), color=BLUE, linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.text(len(parts) - 1 + 0.12, values.mean(axis=0)[-1], "mean of 5 folds", color=INK, fontsize=8.5, va="center")
    ax.text(len(parts) - 1 + 0.12, values[:, -1].min(), "single folds", color=INK_SOFT, fontsize=8.5, va="center")
    ax.set_xticks(range(len(parts)))
    ax.set_xticklabels(parts, color=INK)
    ax.set_xlim(-0.3, len(parts) + 0.6)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("number of parts the clip is divided into", color=INK_SOFT, fontsize=9)
    ax.set_ylabel("validation accuracy, %", color=INK_SOFT, fontsize=9)
    ax.set_title("Temporal statistics: validation accuracy by number of parts", color=INK, fontsize=11, loc="left")
    return _finish(fig, path)


IMPORTANCE_GROUPS = {"tempo": "tempo", "onset_strength": "onset strength", "chroma": "chroma", "tonnetz": "tonnetz",
                     "mfcc": "MFCC", "delta_mfcc": "delta MFCC", "spectral_centroid": "spectral centroid",
                     "spectral_bandwidth": "spectral bandwidth", "spectral_rolloff": "spectral roll-off",
                     "spectral_contrast": "spectral contrast", "zcr": "zero-crossing rate", "rms": "RMS energy"}


def importance_summary(table):
    """Random-forest importance of the aggregate inputs by feature group.

    `table` is results/feature_importance.csv. The importances of a forest sum to 1, so the uniform level
    of one input is 1 / (number of inputs). `ratio` is the mean importance of the inputs of a group
    divided by this level.
    """
    uniform = 1.0 / len(table)
    groups = []
    for key, title in IMPORTANCE_GROUPS.items():
        rows = table[table["group"] == key]
        groups.append({"group": title, "inputs": int(len(rows)), "importance_sum": float(rows["importance"].sum()),
                       "ratio": float(rows["importance"].mean() / uniform)})
    groups.sort(key=lambda row: -row["ratio"])
    top = table.sort_values("importance", ascending=False).head(10)
    return {"inputs": int(len(table)), "uniform": uniform, "forests": int(table["forests"].iloc[0]),
            "top": [{"feature": r.feature, "statistic": r.statistic, "importance": float(r.importance),
                     "ratio": float(r.importance / uniform)} for r in top.itertuples()],
            "groups": groups}


def figure_importance(table, path=None, top=30):
    """The most important aggregate inputs of the random forest, with the uniform level as a line."""
    rows = table.sort_values("importance", ascending=False).head(top).iloc[::-1]
    labels = [f"{r.feature} ({r.statistic})" if r.statistic != "value" else r.feature for r in rows.itertuples()]
    fig, ax = _axes(7.2, 6.6)
    ax.barh(range(len(rows)), rows["importance"], xerr=rows["importance_std"], color=BLUE, height=0.62,
            error_kw={"ecolor": INK_SOFT, "elinewidth": 1})
    ax.axvline(1.0 / len(table), color=INK_SOFT, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(1.0 / len(table), len(rows) - 0.2, " uniform level", color=INK_SOFT, fontsize=8.5, va="bottom")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, color=INK, fontsize=8)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("importance in the random forest (mean of 25 forests, bars: standard deviation)", color=INK_SOFT,
                  fontsize=9)
    ax.set_title(f"The {top} most important of the {len(table)} aggregate statistics", color=INK, fontsize=11,
                 loc="left")
    return _finish(fig, path)


def figure_layers(sweeps, path=None):
    """Validation accuracy of the classifier on every layer of each frozen encoder, one panel per encoder."""
    import matplotlib.pyplot as plt
    names = [n for n in ("ast", "wav2vec2", "hubert", "wavlm", "whisper", "mert", "mert330") if n in sweeps]
    columns = min(4, len(names))
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(2.6 * columns + 0.6, 2.3 * rows + 0.6), dpi=150, sharey=True,
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    top = max(100 * np.asarray(sweeps[n]["val_accuracy"]).max() for n in names)
    for ax, name in zip(axes.ravel(), names):
        values = 100 * np.asarray(sweeps[name]["val_accuracy"])
        mean = values.mean(axis=0)
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_SOFT, length=0, labelsize=8)
        ax.fill_between(range(len(mean)), values.min(axis=0), values.max(axis=0), color=BLUE, alpha=0.12, linewidth=0)
        ax.plot(range(len(mean)), mean, color=BLUE, linewidth=2)
        best = int(mean.argmax())
        ax.scatter([best], [mean[best]], s=40, color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.annotate(f"layer {best}: {mean[best]:.0f}%", (best, mean[best]), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
        ax.axhline(100 * config.CHANCE_LEVEL, color=GREY, linewidth=1)
        ax.yaxis.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.set_ylim(0, top + 12)
        ax.set_title(INPUT_LABELS["ast_layers" if name == "ast" else name].replace(", layer chosen on validation", ""),
                     color=INK, fontsize=9.5, loc="left")
        ax.set_xlabel("layer (0 is the input embedding)", color=INK_SOFT, fontsize=8)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    for row in axes:
        row[0].set_ylabel("validation accuracy, %", color=INK_SOFT, fontsize=8.5)
    fig.suptitle("Frozen encoders: validation accuracy of a small classifier on each layer\n"
                 "(line: mean of the 5 folds, band: range of the folds, grey line: chance)",
                 color=INK, fontsize=10, x=0.01, ha="left")
    return _finish(fig, path)


def figure_kinds(kinds, path=None):
    """Clip accuracy on hummed and on whistled clips, one row per model."""
    rows = kinds.sort_values("whistle").reset_index(drop=True)
    fig, ax = _axes(7.6, 0.4 * len(rows) + 1.5)
    y = np.arange(len(rows))
    ax.hlines(y, 100 * rows[["hum", "whistle"]].min(axis=1), 100 * rows[["hum", "whistle"]].max(axis=1),
              color=GRID, linewidth=3)
    ax.scatter(100 * rows["hum"], y, s=52, color=BLUE, marker="o", edgecolors=SURFACE, linewidths=1.5, zorder=3,
               label="hum")
    ax.scatter(100 * rows["whistle"], y, s=52, color=ORANGE, marker="s", edgecolors=SURFACE, linewidths=1.5, zorder=3,
               label="whistle")
    ax.axvline(100 * config.CHANCE_LEVEL, color=GREY, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([_short(n) for n in rows["experiment"]], color=INK, fontsize=9)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(0, None)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("accuracy on 10 s clips, % (grey line: chance)", color=INK_SOFT, fontsize=9)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_title("Hummed and whistled clips", color=INK, fontsize=11, loc="left", pad=28)
    return _finish(fig, path)


def _finish(fig, path):
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, facecolor=fig.get_facecolor(), metadata={"Software": None})
    return fig


# Text

def percent(value, digits=1):
    return f"{100 * value:.{digits}f}%"


def markdown_model_table(table):
    """The model table as markdown, accuracies in percent."""
    head = ["Model", "Input", "Variant", "Clip accuracy", "95% interval", "Recording accuracy", "95% interval",
            "Validation", "Train"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in table.itertuples():
        spread = f" ± {100 * row.clip_std:.1f}" if row.seeds > 1 else ""
        spread_rec = f" ± {100 * row.recording_std:.1f}" if row.seeds > 1 else ""
        val = percent(row.validation_accuracy) if np.isfinite(row.validation_accuracy) else "not defined"
        train = percent(row.train_accuracy) if np.isfinite(row.train_accuracy) else "not defined"
        lines.append(
            f"| {row.model} | {row.input} | {row.variant or 'plain'} | {percent(row.clip_accuracy)}{spread} | "
            f"{100 * row.clip_ci_low:.1f} to {100 * row.clip_ci_high:.1f} | "
            f"{percent(row.recording_accuracy)}{spread_rec} | "
            f"{100 * row.recording_ci_low:.1f} to {100 * row.recording_ci_high:.1f} | {val} | {train} |")
    return "\n".join(lines)


def markdown_comparisons(table):
    lines = ["| A | B | Accuracy of A minus accuracy of B, points | 95% interval | Interval excludes 0 |",
             "|---|---|---|---|---|"]
    for row in table.itertuples():
        lines.append(f"| `{row.a}` | `{row.b}` | {100 * row.difference:+.1f} | "
                     f"{100 * row.ci_low:+.1f} to {100 * row.ci_high:+.1f} | "
                     f"{'yes' if row.interval_excludes_zero else 'no'} |")
    return "\n".join(lines)


def markdown_checks(table):
    head = ["Model", "Check", "Clip accuracy", "95% interval", "Recording accuracy",
            "Clip accuracy with participant folds"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in table.itertuples():
        reference = row.clip_accuracy_participant_folds
        lines.append(
            f"| {row.model} | {row.check} | {percent(row.clip_accuracy)} ± {100 * row.clip_std:.1f} | "
            f"{100 * row.clip_ci_low:.1f} to {100 * row.clip_ci_high:.1f} | {percent(row.recording_accuracy)} ± "
            f"{100 * row.recording_std:.1f} | {percent(reference) if np.isfinite(reference) else 'not run'} |")
    return "\n".join(lines)


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.write("\n")
