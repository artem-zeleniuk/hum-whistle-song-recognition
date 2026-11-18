"""Train the sequence models and the classifier on the pretrained embeddings.

    python scripts/run_neural.py                            # every experiment
    python scripts/run_neural.py --only cnn1d__melody_full__none --processes 1

Experiments (names are <model>__<input>__<augmentation arm>):
  1D CNN on three sets of frame-level features and with the two augmentation arms, 2D CNN, bidirectional
  LSTM, the two sequence models on the pitch contour, and a small classifier on the frozen embeddings
  of pretrained audio encoders (for these the encoder layer is chosen per fold on the validation fold).
With --checks the script runs the protocol checks instead.
Each experiment is trained on 5 folds with 5 seeds and saved to results/predictions/<name>.npz.
"""

import argparse
import json
import multiprocessing
import os

import numpy as np
import torch

from _common import add_common_arguments, log

from humrec import config, evaluation, experiments


def _run(task):
    cache_dir, name, fold, seed, device, max_epochs, layer = task
    torch.set_num_threads(2)
    if device == "cuda" and torch.cuda.device_count() > 1:
        # several GPUs: every worker process keeps to one of them
        worker = multiprocessing.current_process()._identity
        if worker:
            device = f"cuda:{(worker[0] - 1) % torch.cuda.device_count()}"
    return experiments.run_neural(cache_dir, name, fold, seed, device=device, max_epochs=max_epochs, layer=layer)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", default=None, help="names of the experiments to run")
    parser.add_argument("--checks", action="store_true", help="run the protocol checks instead of the experiments")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(config.SEEDS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--processes", type=int, default=4, help="trainings that share the device")
    parser.add_argument("--max-epochs", type=int, default=None, help="upper limit of epochs (for quick tests)")
    add_common_arguments(parser)
    args = parser.parse_args()
    cache = experiments.Cache(args.cache_dir)
    n_clips = len(cache.clips)
    out_dir = os.path.join(args.results_dir, "predictions")
    names = args.only or list(experiments.NEURAL)
    if args.checks:
        names = [n for n, c in experiments.CHECKS.items() if c["base"] in experiments.NEURAL]
    log(f"{n_clips} clips, {len(names)} experiments, device {args.device}, {args.processes} trainings in parallel")

    with multiprocessing.get_context("spawn").Pool(args.processes) as pool:
        for name in names:
            check = experiments.CHECKS.get(name)
            spec = dict(experiments.NEURAL[check["base"]], check=check["check"]) if check else experiments.NEURAL[name]
            needed = {"ast": "ast_cls", "contour": "f0", "contour_wide": "f0_wide"}.get(
                spec.get("source", spec["input"]))
            if "encoder" in spec:
                needed = f"emb_{spec['encoder']}"
            if needed and not cache.has(needed):
                log(f"{name}: skipped, {needed}.npy is not in the cache")
                continue
            info = {"experiment": name, **spec, "seeds": args.seeds}
            layer_of_fold = {k: None for k in range(config.N_FOLDS)}
            if "encoder" in spec:
                # one training per fold and layer with the first seed, the layer with the best validation
                # accuracy of a fold is then used for all seeds of that fold
                layers = list(range(cache.n_layers(spec["encoder"])))
                sweep = pool.map(_run, [(args.cache_dir, name, k, args.seeds[0], args.device, args.max_epochs, layer)
                                        for k in range(config.N_FOLDS) for layer in layers], chunksize=1)
                table = np.array([r["val_accuracy"] for r in sweep]).reshape(config.N_FOLDS, len(layers))
                layer_of_fold = {k: int(table[k].argmax()) for k in range(config.N_FOLDS)}
                info["layer_sweep"] = {"layers": layers, "val_accuracy": table.tolist(), "chosen": layer_of_fold}
                sweep_path = os.path.join(args.results_dir, "layer_sweep.json")
                saved = {}
                if os.path.exists(sweep_path):
                    with open(sweep_path) as handle:
                        saved = json.load(handle)
                saved[spec["encoder"]] = info["layer_sweep"]
                with open(sweep_path, "w") as handle:
                    json.dump(saved, handle, indent=1)
                log(f"{name}: validation accuracy per layer {np.round(table.mean(axis=0), 3).tolist()}, "
                    f"chosen per fold {layer_of_fold}")
            tasks = [(args.cache_dir, name, k, seed, args.device, args.max_epochs, layer_of_fold[k])
                     for seed in args.seeds for k in range(config.N_FOLDS)]
            results = pool.map(_run, tasks, chunksize=1)
            proba, val, train, runs = experiments.assemble(results, n_clips, args.seeds)
            info["runs"] = runs
            evaluation.save_predictions(out_dir, name, proba, val, train, info)
            accuracy = (proba.argmax(axis=2) == cache.labels[None, :]).mean(axis=1)
            epochs = [r["epochs"] for r in runs]
            log(f"{name}: test clip accuracy per seed {np.round(accuracy, 4).tolist()}, validation {val.mean():.4f}, "
                f"train {train.mean():.4f}, epochs {min(epochs)} to {max(epochs)}")


if __name__ == "__main__":
    main()
