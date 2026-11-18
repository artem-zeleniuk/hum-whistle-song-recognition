"""Train variants of the experiments on fixed-length inputs: two-stage and noisy-data methods.

    python scripts/run_variants.py --only xgboost__aggregate_two_stage__none mlp__hubert_mixup__none

Variants (names are <model>__<input>_<variant>__none, the list is in humrec/experiments.py):
  two_stage  a first model predicts hum or whistle, then one song model per type
  kind_norm  one song model, inputs standardised separately for hums and for whistles
  mixup      label smoothing and mixup while the classifier on the embeddings is trained
  pruned     the least recognisable training recordings are left out, the share is chosen on validation
For the experiments on pretrained embeddings the encoder layer of each fold is the one chosen by the plain
experiment (results/layer_sweep.json). Each variant is trained on 5 folds with 5 seeds.
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
    cache_dir, name, fold, seed, device, threads, layer, max_epochs = task
    torch.set_num_threads(2)
    if device == "cuda" and torch.cuda.device_count() > 1:
        # several GPUs: every worker process keeps to one of them
        worker = multiprocessing.current_process()._identity
        if worker:
            device = f"cuda:{(worker[0] - 1) % torch.cuda.device_count()}"
    return experiments.run_variant(cache_dir, name, fold, seed, device=device, n_jobs=threads, layer=layer,
                                   max_epochs=max_epochs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", default=None, help="names of the variants to run (default: all)")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(config.SEEDS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4, help="threads of one classical model")
    parser.add_argument("--max-epochs", type=int, default=None, help="upper limit of epochs (for quick tests)")
    add_common_arguments(parser)
    args = parser.parse_args()
    cache = experiments.Cache(args.cache_dir)
    out_dir = os.path.join(args.results_dir, "predictions")
    sweep_path = os.path.join(args.results_dir, "layer_sweep.json")
    sweeps = {}
    if os.path.exists(sweep_path):
        with open(sweep_path) as handle:
            sweeps = json.load(handle)
    names = args.only or list(experiments.VARIANTS)
    log(f"{len(cache.clips)} clips, {len(names)} variants, device {args.device}")
    with multiprocessing.get_context("spawn").Pool(args.processes) as pool:
        for name in names:
            variant = experiments.VARIANTS[name]
            spec = {**experiments.CLASSICAL, **experiments.NEURAL}[variant["base"]]
            layer_of_fold = {k: None for k in range(config.N_FOLDS)}
            if "encoder" in spec:
                if spec["encoder"] not in sweeps:
                    log(f"{name}: skipped, run {variant['base']} first (it chooses the layers)")
                    continue
                layer_of_fold = {int(k): int(v) for k, v in sweeps[spec["encoder"]]["chosen"].items()}
            tasks = [(args.cache_dir, name, k, seed, args.device, args.threads, layer_of_fold[k], args.max_epochs)
                     for seed in args.seeds for k in range(config.N_FOLDS)]
            results = pool.map(_run, tasks, chunksize=1)
            proba, val, train, runs = experiments.assemble(results, len(cache.clips), args.seeds)
            info = {"experiment": name, **spec, **variant, "seeds": args.seeds, "runs": runs}
            evaluation.save_predictions(out_dir, name, proba, val, train, info)
            accuracy = (proba.argmax(axis=2) == cache.labels[None, :]).mean(axis=1)
            extra = ""
            if "kind_accuracy_test" in runs[0]:
                share = np.mean([r["kind_accuracy_test"] for r in runs])
                extra = f", hum or whistle predicted right for {share:.4f} of the test clips"
            if "pruned_share" in runs[0]:
                extra = f", pruned shares chosen {sorted(set(r['pruned_share'] for r in runs))}"
            log(f"{name}: test clip accuracy per seed {np.round(accuracy, 4).tolist()}, "
                f"validation {val.mean():.4f}{extra}")


if __name__ == "__main__":
    main()
