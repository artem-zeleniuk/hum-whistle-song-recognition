"""Build the tables and the figures from the saved predictions.

    python scripts/make_report.py

Reads results/clips.csv, results/recordings.csv, results/predictions/*.npz, results/temporal_sweep.json,
results/layer_sweep.json, results/augmentation_diagnostic.csv, results/duplicate_check.json,
results/feature_importance.csv and, for the experiments that were trained a second time,
results/repeat/predictions/*.npz. Writes the ensembles to
results/predictions, the tables to results/tables, all numbers to results/metrics.json and the figures to
figures/. Prints the table of all experiments. Needs neither the audio nor a GPU and runs in a few seconds.
"""

import argparse
import json
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from _common import ROOT, log  # noqa: E402

from humrec import report  # noqa: E402

# experiments whose accuracy is reported separately for hummed and for whistled clips and for full and for
# padded clips, next to the final model and the ensemble of the two strongest families
KIND_MODELS = ["xgboost__aggregate__none", "xgboost__temporal__none", "cnn1d__melody_full__none",
               "cnn1d__melody_full_transpose__none", "keynet__melody_full__none", "cnn1d__contour__none",
               "cnn1d__contour_wide__none", "knn__contour_dtw__none", "mlp__mert__none", "finetune__mert__none"]
# experiments with a confusion matrix, next to the final model
CONFUSION_MODELS = ["xgboost__aggregate__none", "cnn1d__contour__none", "ensemble__top2"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    parser.add_argument("--figures-dir", default=os.path.join(ROOT, "figures"))
    args = parser.parse_args()
    tables_dir = os.path.join(args.results_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    ensembles = report.build_ensembles(args.results_dir)
    log(f"ensembles written: {ensembles}")
    clips, predictions = report.load_all(args.results_dir)
    log(f"{len(predictions)} experiments, {len(clips)} clips")

    models = report.model_table(clips, predictions)
    comparisons = report.comparison_table(clips, predictions)
    comparisons_recording = report.comparison_table(clips, predictions, level="recording")
    final = report.final_model(models)
    kind_models = list(dict.fromkeys(KIND_MODELS + [final, "ensemble__top2"]))
    kinds = report.kind_table(clips, predictions, kind_models)
    checks = report.checks_table(clips, predictions)
    repeated = report.repeat_table(clips, predictions, args.results_dir)
    selection = report.selection_check(clips, predictions)
    padded = report.padded_table(clips, predictions, kind_models)
    data = report.data_summary(args.results_dir)
    models.to_csv(os.path.join(tables_dir, "models.csv"), index=False, float_format="%.6f")
    comparisons.to_csv(os.path.join(tables_dir, "comparisons_clip.csv"), index=False, float_format="%.6f")
    comparisons_recording.to_csv(os.path.join(tables_dir, "comparisons_recording.csv"), index=False,
                                 float_format="%.6f")
    kinds.to_csv(os.path.join(tables_dir, "hum_whistle.csv"), index=False, float_format="%.6f")
    if len(checks):
        checks.to_csv(os.path.join(tables_dir, "protocol_checks.csv"), index=False, float_format="%.6f")
    if len(repeated):
        repeated.to_csv(os.path.join(tables_dir, "repeated_runs.csv"), index=False, float_format="%.6f")
    selection.to_csv(os.path.join(tables_dir, "selection_check.csv"), index=False, float_format="%.6f")
    padded.to_csv(os.path.join(tables_dir, "padded_clips.csv"), index=False, float_format="%.6f")

    metrics = {"data": data, "models": models.set_index("experiment").to_dict(orient="index"),
               "comparisons_clip": comparisons.to_dict(orient="records"),
               "comparisons_recording": comparisons_recording.to_dict(orient="records"),
               "hum_whistle": kinds.to_dict(orient="records"),
               "protocol_checks": checks.to_dict(orient="records"),
               "repeated_runs": repeated.to_dict(orient="records"),
               "selection_check": selection.to_dict(orient="records"),
               "padded_clips": padded.to_dict(orient="records")}
    importance_path = os.path.join(args.results_dir, "feature_importance.csv")
    if os.path.exists(importance_path):
        importance = pd.read_csv(importance_path)
        metrics["feature_importance"] = report.importance_summary(importance)
        pd.DataFrame(metrics["feature_importance"]["groups"]).to_csv(
            os.path.join(tables_dir, "feature_importance_groups.csv"), index=False, float_format="%.6f")
        report.figure_importance(importance, os.path.join(args.figures_dir, "feature_importance.png"))
    duplicates_path = os.path.join(args.results_dir, "duplicate_check.json")
    if os.path.exists(duplicates_path):
        with open(duplicates_path) as handle:
            metrics["duplicate_check"] = json.load(handle)

    report.figure_overview(models, os.path.join(args.figures_dir, "accuracy_by_model.png"))
    report.figure_augmentation(models, os.path.join(args.figures_dir, "augmentation.png"))
    if len(kinds):
        report.figure_kinds(kinds, os.path.join(args.figures_dir, "hum_and_whistle.png"))
    metrics["final_model"] = final
    confusions = {}
    for name in dict.fromkeys(CONFUSION_MODELS + [final]):
        if name not in predictions:
            continue
        matrix = report.confusion(predictions[name]["proba"], clips)
        confusions[name] = matrix.tolist()
        pd.DataFrame(matrix, index=report.config.SONGS, columns=report.config.SONGS).to_csv(
            os.path.join(tables_dir, f"confusion_{name}.csv"))
        report.figure_confusion(matrix, "Confusion matrix: " + report._short(name),
                                os.path.join(args.figures_dir, f"confusion_{name}.png"))
    metrics["confusion"] = confusions
    sweep_path = os.path.join(args.results_dir, "temporal_sweep.json")
    sweep = None
    if os.path.exists(sweep_path):
        with open(sweep_path) as handle:
            sweep = json.load(handle)
        report.figure_parts(sweep, os.path.join(args.figures_dir, "temporal_parts.png"))
    layers_path = os.path.join(args.results_dir, "layer_sweep.json")
    layers = None
    if os.path.exists(layers_path):
        with open(layers_path) as handle:
            layers = json.load(handle)
        report.figure_layers(layers, os.path.join(args.figures_dir, "encoder_layers.png"))
    for key, file_name in (("pitch_tracking", "pitch_summary.json"),
                           ("pitch_tracking_wide", "pitch_summary_wide.json")):
        pitch_path = os.path.join(args.results_dir, file_name)
        if os.path.exists(pitch_path):
            with open(pitch_path) as handle:
                metrics[key] = json.load(handle)
    metrics["ensemble_best_members"] = report.best_members(args.results_dir)
    metrics["ensemble_top2_members"] = report.best_members(args.results_dir, top=2)
    report.write_json(os.path.join(args.results_dir, "metrics.json"), metrics)

    log(f"final model (highest validation accuracy): {final}")
    print(models[["experiment", "clip_accuracy", "clip_std", "recording_accuracy", "validation_accuracy",
                  "train_accuracy"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
