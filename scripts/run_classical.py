"""Train the models on fixed-length feature vectors with participant-disjoint cross-validation.

    python scripts/run_classical.py                         # every experiment
    python scripts/run_classical.py --only xgboost__aggregate__none

Experiments (names are <model>__<input>__<augmentation arm>):
  random_forest, svm, xgboost, lightgbm on the aggregate statistics, each without augmentation and with
  the two augmentation arms, XGBoost on the temporal statistics with feature selection (the number of
  parts is chosen per fold on the validation fold), the same with XGBoost settings found by a random
  search inside the training folds, and XGBoost on the pitch-contour statistics.
With --checks the script runs the protocol checks instead (shuffled labels, folds that ignore the
participants).
Each experiment is trained on 5 folds with 5 seeds and saved to results/predictions/<name>.npz.
"""

import argparse
import json
import os

import numpy as np
from joblib import Parallel, delayed

from _common import add_common_arguments, log

from humrec import config, evaluation, experiments


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", default=None, help="names of the experiments to run")
    parser.add_argument("--checks", action="store_true", help="run the protocol checks instead of the experiments")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(config.SEEDS))
    parser.add_argument("--threads", type=int, default=4, help="threads of one model")
    parser.add_argument("--parts", nargs="*", type=int, default=list(experiments.PARTS_GRID))
    parser.add_argument("--tune-iterations", type=int, default=30, help="settings drawn in the XGBoost search")
    add_common_arguments(parser)
    args = parser.parse_args()
    cache = experiments.Cache(args.cache_dir)
    n_clips = len(cache.clips)
    out_dir = os.path.join(args.results_dir, "predictions")
    names = args.only or list(experiments.CLASSICAL)
    if args.checks:
        names = [n for n, c in experiments.CHECKS.items() if c["base"] in experiments.CLASSICAL]
    jobs = max(1, args.workers // args.threads)
    folds = range(config.N_FOLDS)
    log(f"{n_clips} clips, {len(names)} experiments, {jobs} models in parallel with {args.threads} threads each")

    for name in names:
        check = experiments.CHECKS.get(name)
        spec = experiments.CLASSICAL[name] if not check else dict(experiments.CLASSICAL[check["base"]],
                                                                  check=check["check"])
        if spec["input"] == "contour" and not cache.has("f0"):
            log(f"{name}: skipped, no pitch contours in the cache")
            continue
        info = {"experiment": name, **spec, "seeds": args.seeds}
        parts_of_fold = {k: None for k in folds}
        if spec["input"] == "temporal_tuned":
            sweep_path = os.path.join(args.results_dir, "temporal_sweep.json")
            if not os.path.exists(sweep_path):
                log(f"{name}: skipped, run xgboost__temporal__none first (it writes temporal_sweep.json)")
                continue
            with open(sweep_path) as handle:
                chosen = {int(k): int(v) for k, v in json.load(handle)["chosen"].items()}
            per_fold = Parallel(n_jobs=len(folds))(
                delayed(experiments.run_tuned)(args.cache_dir, name, k, args.seeds, max(1, args.workers // len(folds)),
                                               chosen[k], args.tune_iterations) for k in folds)
            results = [r for fold_results in per_fold for r in fold_results]
            proba, val, train, runs = experiments.assemble(results, n_clips, args.seeds)
            info["runs"] = runs
            evaluation.save_predictions(out_dir, name, proba, val, train, info)
            accuracy = (proba.argmax(axis=2) == cache.labels[None, :]).mean(axis=1)
            log(f"{name}: test clip accuracy per seed {np.round(accuracy, 4).tolist()}, "
                f"validation {val.mean():.4f}, train {train.mean():.4f}")
            continue
        if spec["input"] == "temporal":
            sweep = Parallel(n_jobs=jobs)(
                delayed(experiments.run_classical)(args.cache_dir, name, k, args.seeds[0], args.threads, n, True)
                for k in folds for n in args.parts)
            table = np.array([r["val_accuracy"] for r in sweep]).reshape(len(folds), len(args.parts))
            train_table = np.array([r["train_accuracy"] for r in sweep]).reshape(len(folds), len(args.parts))
            parts_of_fold = {k: int(args.parts[int(table[k].argmax())]) for k in folds}
            info["sweep"] = {"parts": args.parts, "val_accuracy": table.tolist(),
                             "train_accuracy": train_table.tolist(), "chosen": parts_of_fold}
            with open(os.path.join(args.results_dir, "temporal_sweep.json"), "w") as handle:
                json.dump(info["sweep"], handle, indent=1)
            log(f"{name}: validation accuracy per number of parts "
                f"{dict(zip(args.parts, np.round(table.mean(axis=0), 4).tolist()))}, chosen per fold {parts_of_fold}")
        results = Parallel(n_jobs=jobs)(
            delayed(experiments.run_classical)(args.cache_dir, name, k, seed, args.threads, parts_of_fold[k])
            for seed in args.seeds for k in folds)
        proba, val, train, runs = experiments.assemble(results, n_clips, args.seeds)
        info["runs"] = runs
        evaluation.save_predictions(out_dir, name, proba, val, train, info)
        accuracy = (proba.argmax(axis=2) == cache.labels[None, :]).mean(axis=1)
        log(f"{name}: test clip accuracy per seed {np.round(accuracy, 4).tolist()}, "
            f"validation {val.mean():.4f}, train {train.mean():.4f}")


if __name__ == "__main__":
    main()
