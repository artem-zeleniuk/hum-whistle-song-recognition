"""The list of experiments and the code that runs one model on one fold with one seed."""

import os

import numpy as np
import pandas as pd

from . import classical, config, data, features, neural, pretrained

ARMS = ("none",) + tuple(config.AUGMENTATION_ARMS)
PARTS_GRID = (4, 5, 6, 8, 10, 12, 16, 20)   # numbers of temporal parts compared on the validation fold

# name -> settings. Names are <model>__<input>__<augmentation arm>.
CLASSICAL = {f"{model}__aggregate__{arm}": {"model": model, "input": "aggregate", "arm": arm}
             for model in classical.MODELS for arm in ARMS}
CLASSICAL["xgboost__temporal__none"] = {"model": "xgboost", "input": "temporal", "arm": "none"}
CLASSICAL["xgboost__temporal_tuned__none"] = {"model": "xgboost", "input": "temporal_tuned", "arm": "none"}
CLASSICAL["xgboost__contour__none"] = {"model": "xgboost", "input": "contour", "arm": "none"}

NEURAL = {
    "cnn1d__frames__none": {"model": "cnn1d", "input": "frames", "arm": "none"},
    "cnn2d__melody_full__none": {"model": "cnn2d", "input": "melody_full", "arm": "none"},
    "bilstm__frames__none": {"model": "bilstm", "input": "frames", "arm": "none"},
    "cnn1d__contour__none": {"model": "cnn1d", "input": "contour", "arm": "none"},
    "bilstm__contour__none": {"model": "bilstm", "input": "contour", "arm": "none"},
    "mlp__ast__none": {"model": "mlp", "input": "ast", "arm": "none"},
    # the two contour models again, trained with random changes of tempo (the melody stays the same)
    "cnn1d__contour_stretch__none": {"model": "cnn1d", "input": "contour_stretch", "source": "contour",
                                     "arm": "none", "options": {"stretch": 0.25}},
    "bilstm__contour_stretch__none": {"model": "bilstm", "input": "contour_stretch", "source": "contour",
                                      "arm": "none", "options": {"stretch": 0.25}},
}
# The grid of the chroma networks: two inputs (`melody` is chroma, onset strength and tempo, `melody_full`
# adds the frame-to-frame differences of the chroma), four augmentation arms and five rows:
#   cnn1d                    the plain 1D CNN
#   cnn1d, <input>_transpose the 1D CNN trained with random changes of key (rotations of the 12 pitch classes)
#   cnn1d, <input>_stretch   the 1D CNN trained with random changes of tempo
#   keynet                   the CNN that is invariant to the key by construction
#   keynet, <input>_stretch  the same trained with random changes of tempo
GRID_ARMS = ("none",) + tuple(config.AUGMENTATION_ARMS) + tuple(config.COMBINED_ARMS)
for _input in ("melody", "melody_full"):
    for _arm in GRID_ARMS:
        NEURAL[f"cnn1d__{_input}__{_arm}"] = {"model": "cnn1d", "input": _input, "arm": _arm}
        NEURAL[f"cnn1d__{_input}_transpose__{_arm}"] = {
            "model": "cnn1d", "input": f"{_input}_transpose", "source": _input, "arm": _arm,
            "options": {"transpose": True}}
        NEURAL[f"keynet__{_input}__{_arm}"] = {"model": "keynet", "input": _input, "arm": _arm}
        for _model in ("cnn1d", "keynet"):
            NEURAL[f"{_model}__{_input}_stretch__{_arm}"] = {
                "model": _model, "input": f"{_input}_stretch", "source": _input, "arm": _arm,
                "options": {"stretch": 0.25}}
# which part of the gain of an augmentation arm comes from which transform, and how much from the mere
# number of training clips (the `repeat` arm adds unchanged copies)
for _arm in ("conservative_repeat", "conservative_noise", "conservative_pitch", "strong_noise", "strong_pitch",
             "strong_stretch"):
    NEURAL[f"cnn1d__melody_full__{_arm}"] = {"model": "cnn1d", "input": "melody_full", "arm": _arm}
# two follow-up questions, each on three cells of the grid:
#   keynet12         the key-invariant network with convolutions over the whole circle of pitch classes
#   transpose_long   random key changes with a limit of 300 epochs instead of 100
for _input, _arm in (("melody", "none"), ("melody_full", "none"), ("melody", "strong")):
    NEURAL[f"keynet12__{_input}__{_arm}"] = {"model": "keynet12", "input": _input, "arm": _arm}
    NEURAL[f"cnn1d__{_input}_transpose_long__{_arm}"] = {
        "model": "cnn1d", "input": f"{_input}_transpose_long", "source": _input, "arm": _arm,
        "options": {"transpose": True, "max_epochs": 300}}

# The same small classifier on the embeddings of other frozen pretrained encoders. Each encoder gives one
# embedding per layer, and the layer is chosen per fold on the validation fold.
PROBES = {("mlp__ast_layers__none" if name == "ast" else f"mlp__{name}__none"):
          {"model": "mlp", "input": "ast_layers" if name == "ast" else name, "arm": "none", "encoder": name}
          for name in pretrained.ENCODERS}
NEURAL.update(PROBES)

# the contour from a pitch tracker that searches up to 4186 Hz, for the high whistles
NEURAL["cnn1d__contour_wide__none"] = {"model": "cnn1d", "input": "contour_wide", "arm": "none"}
# the LSTM with the mean of its outputs over time instead of its final states, and a larger learning rate
for _input in ("contour", "melody", "frames"):
    NEURAL[f"bilstm_pool__{_input}__none"] = {"model": "bilstm_pool", "input": _input, "arm": "none",
                                              "options": {"lr": 2e-3}}

# Variants of an experiment on fixed-length inputs (the aggregate statistics or one embedding per clip).
# Names are <model>__<input>_<variant>__none.
#   two_stage   a first model predicts hum or whistle, then one song model per type
#   kind_norm   one song model, the inputs are standardised separately for hums and for whistles
#   mixup       (networks only) label smoothing 0.1 and mixup 0.4 during training
#   pruned      (networks only) the training recordings that a model trained on the other training
#               participants finds least recognisable are left out, the share is chosen on validation
VARIANT_NAMES = ("two_stage", "kind_norm", "mixup", "pruned")
PRUNE_SHARES = (0.0, 0.1, 0.2, 0.3)
VARIANTS = {}
for _base in ["xgboost__aggregate__none", "mlp__ast__none"] + list(PROBES):
    _model, _input, _ = _base.split("__")
    for _variant in VARIANT_NAMES:
        if _model == "xgboost" and _variant in ("mixup", "pruned"):
            continue
        VARIANTS[f"{_model}__{_input}_{_variant}__none"] = {"base": _base, "variant": _variant}

# Checks of the protocol, run for the best classical model on hand-made features, for the classifier on
# the pretrained embeddings and for the two strongest networks. Names are <model>__<input>__<arm>__<check>.
#   shuffled_labels   the training and validation labels are permuted: the accuracy must fall to chance
#   recording_folds   recordings are split at random, so a participant is on both sides of the split
#   clip_folds        clips are split at random, so overlapping clips of one recording are on both sides
CHECK_NAMES = ("shuffled_labels", "recording_folds", "clip_folds")
CHECKS = {}
for _base in ("xgboost__aggregate__none", "mlp__ast__none", "cnn1d__contour__none",
              "cnn1d__melody_full_transpose__none"):
    for _check in CHECK_NAMES:
        CHECKS[f"{_base}__{_check}"] = {"base": _base, "check": _check}

# Ensembles average the class probabilities of their members. The members are fixed here, before any
# test result is seen.
ENSEMBLES = {
    "ensemble__features": ("xgboost__aggregate__none", "xgboost__temporal__none", "cnn1d__melody_full__none"),
    "ensemble__all": ("xgboost__aggregate__none", "xgboost__temporal__none", "cnn1d__melody_full__none",
                      "bilstm__frames__none", "xgboost__contour__none", "mlp__ast__none"),
}


class Cache:
    """Read access to the arrays written by the extraction scripts."""

    def __init__(self, folder, check=None):
        self.folder = folder
        self.clips = pd.read_csv(os.path.join(folder, "clips.csv"))
        self.labels = self.clips["label"].to_numpy()
        self.fold = self.clips["fold"].to_numpy()
        self.check = check
        if check == "recording_folds":
            first = self.clips.drop_duplicates("recording")
            by_recording = dict(zip(first["recording"], data.assign_random_folds(first["label"])))
            self.fold = self.clips["recording"].map(by_recording).to_numpy()
        elif check == "clip_folds":
            self.fold = data.assign_random_folds(self.labels)
        elif check not in (None, "shuffled_labels"):
            raise ValueError(check)

    def _array(self, name):
        return np.load(os.path.join(self.folder, name + ".npy"), mmap_mode="r")

    def has(self, name):
        return os.path.exists(os.path.join(self.folder, name + ".npy"))

    def frames(self, arm="none"):
        return self._array(f"frames_{arm}"), self._array(f"tempo_{arm}")

    def parent(self, arm):
        """For every augmented clip of an arm, the index of the original clip it was made from."""
        return np.asarray(self._array(f"parent_{arm}"))

    def embedding(self, encoder, layer):
        """Embedding of every clip from one layer of a pretrained encoder, shape (clips, dimensions)."""
        return np.asarray(self._array(f"emb_{encoder}")[:, layer, :], dtype=np.float32)

    def n_layers(self, encoder):
        return self._array(f"emb_{encoder}").shape[1]

    def model_input(self, name, arm="none", n_parts=None):
        """Model input for all clips of one arm (originals for `none`, the added clips otherwise)."""
        if isinstance(name, tuple):
            return self.embedding(*name)
        if name == "ast":
            return np.asarray(self._array("ast_cls"), dtype=np.float32)
        if name in ("contour", "contour_wide"):
            f0 = self._array("f0" if name == "contour" else "f0_wide")
            return np.stack([features.contour_sequence(row) for row in f0])
        if name == "contour_statistics":
            f0 = self._array("f0")
            return np.stack([features.contour_statistics(row) for row in f0])
        frames, tempos = self.frames(arm)
        frames, tempos = np.asarray(frames), np.asarray(tempos)
        if name == "aggregate":
            return features.aggregate(frames, tempos)
        if name == "temporal":
            return features.temporal_statistics(frames, tempos, n_parts)
        if name == "frames":
            return features.sequence_input(frames, tempos, "all")
        return features.sequence_input(frames, tempos, name)

    def split(self, name, arm, fold, n_parts=None):
        """Training, validation and test data of one test fold.

        Returns (x_train, y_train, x_val, y_val, x_test, test_index). Augmented clips join the training
        set only if their original clip is in the training folds. Validation and test clips are never
        augmented.
        """
        train, val, test = data.fold_roles(self.fold, fold)
        x = self.model_input(name, "none", n_parts)
        x_train, y_train = x[train], self.labels[train]
        if arm != "none":
            if isinstance(name, tuple) or name in ("ast", "contour", "contour_wide", "contour_statistics"):
                raise ValueError(f"the input {name} exists for the original clips only, not for the arm {arm}")
            # `conservative_pitch` uses only the pitch-shifted clips of the conservative arm, and
            # `conservative_repeat` adds the unchanged original of every clip of the arm instead
            arm, _, part = arm.partition("_")
            parent = self.parent(arm)
            keep = train[parent]
            if part not in ("", "repeat"):
                keep = keep & (np.asarray(self._array(f"kind_{arm}")) == part)
            extra = x[parent[keep]] if part == "repeat" else self.model_input(name, arm, n_parts)[keep]
            x_train = np.concatenate([x_train, extra])
            y_train = np.concatenate([y_train, self.labels[parent[keep]]])
        return x_train, y_train, x[val], self.labels[val], x[test], np.flatnonzero(test)


def _shuffle(y_train, y_val, fold, seed):
    """Permute the training and the validation labels (the shuffled-label check)."""
    rng = np.random.default_rng([seed, fold, 2026])
    return rng.permutation(y_train), rng.permutation(y_val)


def run_classical(cache_dir, name, fold, seed, n_jobs=1, n_parts=None, sweep=False):
    """Train one classical experiment on one fold with one seed. Returns a result dict."""
    check = CHECKS[name]["check"] if name in CHECKS else None
    spec = CLASSICAL[CHECKS[name]["base"]] if name in CHECKS else CLASSICAL[name]
    cache = Cache(cache_dir, check)
    source = "contour_statistics" if spec["input"] == "contour" else spec["input"]
    x_train, y_train, x_val, y_val, x_test, test_index = cache.split(source, spec["arm"], fold, n_parts)
    if check == "shuffled_labels":
        y_train, y_val = _shuffle(y_train, y_val, fold, seed)
    details = {}
    scale = True
    if spec["input"] == "temporal" and not sweep:
        x_train, x_val, x_test, details = classical.select_features(x_train, y_train, x_val, x_test, n_jobs=n_jobs)
        details["n_parts"] = n_parts
        scale = False
    out = classical.fit_predict(spec["model"], seed, x_train, y_train, x_val, y_val, x_test,
                                n_jobs=n_jobs, sweep=sweep, scale=scale)
    return _result(name, fold, seed, out, y_val, test_index, len(y_train),
                   dict(details, rounds=out["rounds"], n_features=int(x_train.shape[1])))


def run_neural(cache_dir, name, fold, seed, device="cpu", max_epochs=None, layer=None):
    """Train one network experiment on one fold with one seed. Returns a result dict.

    `layer` selects the encoder layer for the experiments on pretrained embeddings (PROBES).
    """
    check = CHECKS[name]["check"] if name in CHECKS else None
    spec = NEURAL[CHECKS[name]["base"]] if name in CHECKS else NEURAL[name]
    cache = Cache(cache_dir, check)
    source = (spec["encoder"], layer) if "encoder" in spec else spec.get("source", spec["input"])
    x_train, y_train, x_val, y_val, x_test, test_index = cache.split(source, spec["arm"], fold)
    if check == "shuffled_labels":
        y_train, y_val = _shuffle(y_train, y_val, fold, seed)
    key_free = spec["model"].startswith("keynet") or spec.get("options", {}).get("transpose", False)
    pooled = neural.pitch_class_channels(x_train.shape[-1]) if key_free else ()
    x_train, x_val, x_test = neural.standardise(x_train, x_val, x_test, pooled=pooled)
    settings = dict(neural.HEAD_SETTINGS if spec["model"] == "mlp" else neural.SEQUENCE_SETTINGS)
    settings.update(spec.get("options", {}))
    if max_epochs is not None:
        settings["max_epochs"] = max_epochs
    out = neural.fit_network(spec["model"], x_train, y_train, x_val, y_val, x_test, seed, device=device, **settings)
    details = {"epochs": out["epochs"], "best_epoch": out["best_epoch"],
               "val_accuracy_per_epoch": [round(v, 4) for v in out["history"]]}
    if layer is not None:
        details["layer"] = int(layer)
    return _result(name, fold, seed, out, y_val, test_index, len(y_train), details)


def run_tuned(cache_dir, name, fold, seeds, n_jobs=1, n_parts=None, n_iter=30):
    """Temporal statistics with feature selection and tuned XGBoost settings, one fold, all seeds.

    The settings are searched once per fold (seed of the first run) by cross-validation inside the
    training folds, then one model per seed is trained with them. Returns one result dict per seed.
    """
    cache = Cache(cache_dir)
    x_train, y_train, x_val, y_val, x_test, test_index = cache.split("temporal", "none", fold, n_parts)
    train, _, _ = data.fold_roles(cache.fold, fold)
    groups = cache.clips["participant"].to_numpy()[train]
    x_train, x_val, x_test, details = classical.select_features(x_train, y_train, x_val, x_test, n_jobs=n_jobs)
    settings, score = classical.tune_xgboost(x_train, y_train, groups, seed=seeds[0], n_iter=n_iter, n_jobs=n_jobs)
    details = dict(details, n_parts=n_parts, settings=settings, inner_cv_accuracy=score,
                   n_features=int(x_train.shape[1]))
    results = []
    for seed in seeds:
        out = classical.fit_predict_fixed(settings, seed, x_train, y_train, x_val, x_test, n_jobs=n_jobs)
        results.append(_result(name, fold, seed, out, y_val, test_index, len(y_train),
                               dict(details, rounds=out["rounds"])))
    return results


def self_confidence(x_train, y_train, participants, recordings, seed=0):
    """How recognisable every training recording is to a model that never saw its participant.

    A logistic regression is trained on three quarters of the training participants and applied to the
    remaining quarter, four times. The value of a recording is the mean probability that these models
    give to its true song over its clips. Returns one value per training clip (the value of its recording).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler
    true_probability = np.zeros(len(y_train))
    folds = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    for inner_train, inner_test in folds.split(x_train, y_train, participants):
        scaler = StandardScaler().fit(x_train[inner_train])
        model = LogisticRegression(max_iter=2000).fit(scaler.transform(x_train[inner_train]), y_train[inner_train])
        proba = np.zeros((len(inner_test), config.N_CLASSES))
        proba[:, model.classes_] = model.predict_proba(scaler.transform(x_train[inner_test]))
        true_probability[inner_test] = proba[np.arange(len(inner_test)), y_train[inner_test]]
    frame = pd.DataFrame({"recording": recordings, "value": true_probability})
    return frame.groupby("recording")["value"].transform("mean").to_numpy()


def run_variant(cache_dir, name, fold, seed, device="cpu", n_jobs=1, layer=None, max_epochs=None):
    """Train one variant (see VARIANTS) on one fold with one seed. Returns a result dict."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    variant = VARIANTS[name]["variant"]
    spec = {**CLASSICAL, **NEURAL}[VARIANTS[name]["base"]]
    cache = Cache(cache_dir)
    source = (spec["encoder"], layer) if "encoder" in spec else spec["input"]
    x_train, y_train, x_val, y_val, x_test, test_index = cache.split(source, "none", fold)
    train, val, test = data.fold_roles(cache.fold, fold)
    neural_settings = dict(neural.HEAD_SETTINGS)
    if max_epochs is not None:
        neural_settings["max_epochs"] = max_epochs
    details = {}

    def fit(a_train, b_train, a_val, b_val, a_test, **extra):
        if spec["model"] in classical.MODELS:
            return classical.fit_predict(spec["model"], seed, a_train, b_train, a_val, b_val, a_test, n_jobs=n_jobs)
        a_train, a_val, a_test = neural.standardise(a_train, a_val, a_test)
        return neural.fit_network(spec["model"], a_train, b_train, a_val, b_val, a_test, seed, device=device,
                                  **neural_settings, **extra)

    if variant in ("two_stage", "kind_norm"):
        whistle = (cache.clips["kind"] == "whistle").to_numpy().astype(int)
        scaler = StandardScaler().fit(x_train)
        first = LogisticRegression(max_iter=2000).fit(scaler.transform(x_train), whistle[train])
        kind_val, kind_test = first.predict(scaler.transform(x_val)), first.predict(scaler.transform(x_test))
        details["kind_accuracy_test"] = float(np.mean(kind_test == whistle[test]))
        if variant == "kind_norm":
            x_train, x_val, x_test = x_train.copy(), x_val.copy(), x_test.copy()
            for kind in (0, 1):
                own = StandardScaler().fit(x_train[whistle[train] == kind])
                masks = (whistle[train] == kind, kind_val == kind, kind_test == kind)
                for block, mask in zip((x_train, x_val, x_test), masks):
                    if mask.any():
                        block[mask] = own.transform(block[mask])
            out = fit(x_train, y_train, x_val, y_val, x_test)
        else:
            out = {"val": np.full((len(y_val), config.N_CLASSES), 1.0 / config.N_CLASSES, dtype=np.float32),
                   "test": np.full((len(x_test), config.N_CLASSES), 1.0 / config.N_CLASSES, dtype=np.float32)}
            right = 0.0
            for kind in (0, 1):
                rows = whistle[train] == kind
                # songs of this type in the training set, renumbered from 0 (all eight unless the data are tiny)
                present = np.unique(y_train[rows])
                number = np.full(config.N_CLASSES, -1)
                number[present] = np.arange(len(present))
                stop_rows = (kind_val == kind) & np.isin(y_val, present)
                if not stop_rows.any():
                    raise ValueError(f"{name}, fold {fold}: no validation clip was predicted as type {kind}, "
                                     "so its song model cannot be stopped early (the data are too small)")
                targets = np.concatenate([x_val[kind_val == kind], x_test[kind_test == kind]])
                part = fit(x_train[rows], number[y_train[rows]], x_val[stop_rows], number[y_val[stop_rows]], targets)
                proba = np.zeros((len(targets), config.N_CLASSES), dtype=np.float32)
                proba[:, present] = part["test"][:, :len(present)]
                n_val = int((kind_val == kind).sum())
                out["val"][kind_val == kind] = proba[:n_val]
                out["test"][kind_test == kind] = proba[n_val:]
                right += part["train_accuracy"] * rows.sum()
            out["train_accuracy"] = float(right / len(y_train))
    elif variant == "mixup":
        out = fit(x_train, y_train, x_val, y_val, x_test, label_smoothing=0.1, mixup=0.4)
    elif variant == "pruned":
        value = self_confidence(x_train, y_train, cache.clips["participant"].to_numpy()[train],
                                cache.clips["recording"].to_numpy()[train], seed=seed)
        best = None
        for share in PRUNE_SHARES:
            keep = np.ones(len(y_train), dtype=bool)
            for song in range(config.N_CLASSES):
                rows = y_train == song
                keep[rows] = value[rows] >= np.quantile(value[rows], share)
            candidate = fit(x_train[keep], y_train[keep], x_val, y_val, x_test)
            accuracy = float(np.mean(candidate["val"].argmax(axis=1) == y_val))
            if best is None or accuracy > best[0]:
                best = (accuracy, share, candidate)
        out = best[2]
        details["pruned_share"] = best[1]
    else:
        raise ValueError(variant)
    if layer is not None:
        details["layer"] = int(layer)
    return _result(name, fold, seed, out, y_val, test_index, len(y_train), details)


def _result(name, fold, seed, out, y_val, test_index, n_train, details):
    return {
        "name": name, "fold": int(fold), "seed": int(seed), "test_index": test_index,
        "test": out["test"], "n_train": int(n_train),
        "val_accuracy": float(np.mean(out["val"].argmax(axis=1) == y_val)),
        "train_accuracy": out["train_accuracy"], "details": details,
    }


def assemble(results, n_clips, seeds=config.SEEDS, n_folds=config.N_FOLDS):
    """Join the per-fold results of one experiment into arrays over seeds, clips and folds."""
    seeds = list(seeds)
    proba = np.full((len(seeds), n_clips, config.N_CLASSES), np.nan, dtype=np.float32)
    val = np.full((len(seeds), n_folds), np.nan, dtype=np.float32)
    train = np.full((len(seeds), n_folds), np.nan, dtype=np.float32)
    runs = []
    for r in sorted(results, key=lambda r: (r["seed"], r["fold"])):
        s = seeds.index(r["seed"])
        proba[s, r["test_index"]] = r["test"]
        val[s, r["fold"]] = r["val_accuracy"]
        train[s, r["fold"]] = r["train_accuracy"]
        runs.append(dict(r["details"], fold=r["fold"], seed=r["seed"], n_train=r["n_train"]))
    if np.isnan(proba).any():
        raise RuntimeError("some clips have no prediction")
    return proba, val, train, runs
