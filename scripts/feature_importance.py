"""Importance of the 109 aggregate statistics in the random forest.

    python scripts/feature_importance.py

For every fold and every seed the random forest of the experiment random_forest__aggregate__none is
trained on the training folds, and the impurity importance of its 109 inputs is kept (the mean and the
standard deviation over time of the 54 frame-level features, and the tempo). The importances of one
forest sum to 1, so an input that is used like every other has 1/109.

Writes results/feature_importance.csv with one row per input: the feature, its group, the statistic and
the mean and standard deviation of the importance over the 25 trainings. Needs the cache written by
scripts/extract_features.py, no GPU, and takes under a minute.
"""

import argparse
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

from _common import add_common_arguments, log

from humrec import classical, config, experiments


def input_names():
    """(feature, group, statistic) of the 109 aggregate inputs, in their order."""
    rows = []
    for statistic in ("mean", "std"):
        for group, n in config.FEATURE_GROUPS:
            for i in range(n):
                rows.append((group if n == 1 else f"{group}_{i + 1}", group, statistic))
    rows.append(("tempo", "tempo", "value"))
    return rows


def importance_of_one_forest(cache_dir, fold, seed):
    cache = experiments.Cache(cache_dir)
    x_train, y_train, _, _, _, _ = cache.split("aggregate", "none", fold)
    model = classical.build("random_forest", seed)
    model.fit(StandardScaler().fit_transform(x_train), y_train)
    return model.feature_importances_


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="*", type=int, default=list(config.SEEDS))
    add_common_arguments(parser)
    args = parser.parse_args()
    tasks = [(fold, seed) for fold in range(config.N_FOLDS) for seed in args.seeds]
    log(f"{len(tasks)} random forests on the aggregate statistics, {args.workers} worker processes")
    values = np.array(Parallel(n_jobs=args.workers)(
        delayed(importance_of_one_forest)(args.cache_dir, fold, seed) for fold, seed in tasks))
    names = input_names()
    assert values.shape[1] == len(names), (values.shape, len(names))
    table = pd.DataFrame(names, columns=["feature", "group", "statistic"])
    table["importance"] = values.mean(axis=0)
    table["importance_std"] = values.std(axis=0)
    table["forests"] = len(tasks)
    table = table.sort_values("importance", ascending=False).reset_index(drop=True)
    table.to_csv(os.path.join(args.results_dir, "feature_importance.csv"), index=False, float_format="%.6f")
    log(f"uniform level 1/{len(names)} = {1 / len(names):.4f}, the ten most important inputs:")
    print(table.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
