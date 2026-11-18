"""Smoke tests on synthetic audio. They need neither the data set nor a GPU and run in about half a minute.

    python tests/test_smoke.py        or        python -m pytest tests
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from humrec import audio, classical, config, data, evaluation, experiments, features, neural  # noqa: E402


def melody(seconds, base=220.0, sr=config.SAMPLE_RATE, seed=0):
    """A sequence of sine tones, one every half second, with a little noise."""
    rng = np.random.default_rng(seed)
    notes = base * 2.0 ** (rng.integers(0, 12, size=int(seconds * 2)) / 12.0)
    t = np.arange(int(0.5 * sr)) / sr
    wave = np.concatenate([np.sin(2 * np.pi * f * t) for f in notes])
    return (0.3 * wave + 0.001 * rng.standard_normal(len(wave))).astype(np.float32)


def test_file_names():
    row = data.parse_filename("S105_whistle_2_Happy.wav")
    assert row == {"participant": "S105", "kind": "whistle", "take": 2, "song": "Happy"}
    for bad in ("S105_sing_2_Happy.wav", "S105_hum_2_Unknown.wav", "notes.wav"):
        try:
            data.parse_filename(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should be rejected")


def test_clips():
    sr = config.SAMPLE_RATE
    silence = np.zeros(sr, dtype=np.float32)
    # windows start every 5 s, and what is left after the last full window is kept when it is 5 s or longer
    for seconds, expected_clips, last_real in ((23.0, 4, 8.0), (12.0, 2, 7.0), (14.0, 2, 9.0), (7.0, 1, 7.0)):
        samples = np.concatenate([silence, melody(seconds), silence])
        clips = audio.split_into_clips(samples)
        assert len(clips) == expected_clips, (seconds, len(clips))
        assert all(len(clip) == config.CLIP_SAMPLES for _, clip, _ in clips)
        assert abs(clips[-1][2] - last_real) < 0.2, (seconds, clips[-1][2])
        assert abs(clips[0][0] - 1.0) < 0.1          # the leading second of silence is cut
    assert audio.split_into_clips(np.concatenate([silence, melody(3.0), silence])) == []


def test_augmentation_is_repeatable():
    clip = melody(config.CLIP_SECONDS)
    for arm in config.AUGMENTATION_ARMS:
        first = audio.augmented_versions(clip, arm, audio.clip_rng("S1_hum_1_Happy.wav", 0, arm))
        second = audio.augmented_versions(clip, arm, audio.clip_rng("S1_hum_1_Happy.wav", 0, arm))
        assert [k for k, _ in first] == [k for k, _ in second]
        for (_, a), (_, b) in zip(first, second):
            assert a.shape == clip.shape and np.array_equal(a, b)
    combined = audio.augmented_versions(clip, "heavy", audio.clip_rng("S1_hum_1_Happy.wav", 0, "heavy"))
    assert len(combined) == config.COMBINED_ARMS["heavy"]["copies"] and all(c.shape == clip.shape for _, c in combined)
    assert not np.allclose(combined[0][1], combined[1][1])
    drawn = [len(audio.augmented_versions(clip[:2048], "conservative", audio.clip_rng("x.wav", i, "conservative")))
             for i in range(200)]
    assert 0.6 < np.mean(drawn) < 1.0                # two transforms, each with probability 0.4


def test_features():
    clip = melody(config.CLIP_SECONDS)
    frames = features.frame_features(clip)
    assert frames.shape == (config.N_FRAMES, config.N_FEATURES) and np.isfinite(frames).all()
    assert len(config.feature_names()) == config.N_FEATURES
    assert config.feature_slice("chroma") == slice(13, 25) and config.feature_slice("onset_strength") == slice(53, 54)
    bpm = features.tempo(clip)
    assert 30 < bpm < 300
    batch = np.stack([frames, frames])
    tempos = np.array([bpm, bpm])
    assert features.aggregate(batch, tempos).shape == (2, 109)
    assert features.temporal_statistics(batch, tempos, 4).shape == (2, 4 * 54 * 2 + 1)
    assert features.sequence_input(batch, tempos, "all").shape == (2, config.N_FRAMES, 54)
    assert features.sequence_input(batch, tempos, "melody").shape == (2, config.N_FRAMES, 14)
    assert features.sequence_input(batch, tempos, "melody_full").shape == (2, config.N_FRAMES, 26)


def test_pitch_contour_ignores_the_key():
    low = features.pitch_track(melody(config.CLIP_SECONDS, base=220.0, seed=3))
    high = features.pitch_track(melody(config.CLIP_SECONDS, base=330.0, seed=3))
    assert np.isfinite(low).mean() > 0.8
    assert abs(np.nanmedian(high) / np.nanmedian(low) - 1.5) < 0.05
    a, b = features.contour_sequence(low), features.contour_sequence(high)
    assert a.shape == (config.N_FRAMES, 3)
    both = (a[:, 2] > 0) & (b[:, 2] > 0)
    assert np.median(np.abs(a[both, 0] - b[both, 0])) < 0.1    # the same melody a fifth higher
    stats = features.contour_statistics(low)
    assert stats.shape == (len(features.CONTOUR_STATISTIC_NAMES),) and np.isfinite(stats).all()
    assert abs(stats[:3].sum() - 1.0) < 1e-5
    assert not features.contour_statistics(np.full(config.N_FRAMES, np.nan)).any()


def toy_recordings(n_participants=40):
    import pandas as pd
    rows = []
    for p in range(n_participants):
        for song in range(config.N_CLASSES):
            if (p + song) % 2 == 0:
                rows.append({"participant": f"S{p}", "label": song})
    return pd.DataFrame(rows)


def test_folds_keep_participants_apart():
    recordings = toy_recordings()
    fold = data.assign_folds(recordings)
    assert set(fold) == set(range(config.N_FOLDS))
    assert recordings.assign(fold=fold).groupby("participant")["fold"].nunique().max() == 1
    for k in range(config.N_FOLDS):
        train, val, test = data.fold_roles(fold, k)
        assert (train.astype(int) + val + test == 1).all()
        people = [set(recordings["participant"][mask]) for mask in (train, val, test)]
        assert not (people[0] & people[1]) and not (people[0] & people[2]) and not (people[1] & people[2])
    assert np.array_equal(fold, data.assign_folds(recordings))


def separable(n, d, seed):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % config.N_CLASSES
    x = rng.standard_normal((n, d)).astype(np.float32)
    x[np.arange(n), y] += 4.0
    return x, y


def test_classical_models_learn():
    x_train, y_train = separable(400, 20, 0)
    x_val, y_val = separable(160, 20, 1)
    x_test, y_test = separable(160, 20, 2)
    for name in classical.MODELS:
        out = classical.fit_predict(name, 0, x_train, y_train, x_val, y_val, x_test)
        assert out["test"].shape == (160, config.N_CLASSES)
        assert np.allclose(out["test"].sum(axis=1), 1.0, atol=1e-4)
        assert np.mean(out["test"].argmax(axis=1) == y_test) > 0.8, name
    selected = classical.select_features(np.hstack([x_train, x_train[:, :5] * 2.0]), y_train,
                                         np.hstack([x_val, x_val[:, :5] * 2.0]),
                                         np.hstack([x_test, x_test[:, :5] * 2.0]))
    assert selected[3]["after_correlation"] == 20 and selected[0].shape[1] == selected[3]["selected"] <= 20


def test_networks_train():
    rng = np.random.default_rng(0)
    y = np.arange(64) % config.N_CLASSES
    for kind, shape in (("cnn1d", (64, 96, 6)), ("cnn2d", (64, 96, 12)), ("bilstm", (64, 40, 6)), ("mlp", (64, 768))):
        x = rng.standard_normal(shape).astype(np.float32)
        x[np.arange(64), ..., y % shape[-1]] += 3.0
        x_std, = neural.standardise(x)
        out = neural.fit_network(kind, x_std, y, x_std, y, x_std, seed=0, max_epochs=3, batch_size=16)
        assert out["test"].shape == (64, config.N_CLASSES) and np.isfinite(out["test"]).all()
        assert np.allclose(out["test"].sum(axis=1), 1.0, atol=1e-4)
        assert 1 <= out["best_epoch"] <= out["epochs"] <= 3
    lstm = neural.BiLSTMNet(54)
    assert neural.count_parameters(lstm) == 23048
    import torch
    batch = torch.arange(2 * 100 * 3, dtype=torch.float32).reshape(2, 100, 3)
    for factor in (0.8, 1.0, 1.25):
        assert neural.stretch_in_time(batch, factor, np.random.default_rng(0)).shape == batch.shape
    assert torch.equal(neural.stretch_in_time(batch, 1.0, np.random.default_rng(0)), batch)
    sequences = rng.standard_normal((32, 60, 3)).astype(np.float32)
    targets = np.arange(32) % config.N_CLASSES
    regularised = neural.fit_network("cnn1d", sequences, targets, sequences, targets, sequences, seed=0, max_epochs=2,
                                     batch_size=16, stretch=0.25, label_smoothing=0.1)
    assert np.isfinite(regularised["test"]).all()


def test_key_invariance():
    import torch
    rng = np.random.default_rng(1)
    x = torch.as_tensor(rng.standard_normal((4, 64, 26)).astype(np.float32))
    with torch.no_grad():
        networks = ((14, neural.KeyInvariantNet), (26, neural.KeyInvariantNet), (14, neural.KeyInvariantNetFull))
        for channels, network in networks:         # 14: the melody layout, 26: the melody_full layout
            net = network(channels).eval()
            reference = net(x[:, :, :channels])
            for shift in (1, 5, 11):
                moved = x[:, :, :channels].clone()
                for start, stop in neural.pitch_class_channels(channels):
                    moved[:, :, start:stop] = torch.roll(x[:, :, start:stop], shift, dims=2)
                assert torch.allclose(net(moved), reference, atol=1e-5), (channels, shift)
        plain = neural.Conv1dNet(26).eval()
        moved = x.clone()
        moved[:, :, :12] = torch.roll(x[:, :, :12], 5, dims=2)
        assert not torch.allclose(plain(moved), plain(x), atol=1e-5)     # an ordinary network does depend on the key
    moved = neural.transpose_key(x, np.random.default_rng(0))
    assert moved.shape == x.shape and torch.equal(moved[:, :, 24:], x[:, :, 24:])
    assert torch.allclose(moved[:, :, :12].sum(dim=2), x[:, :, :12].sum(dim=2), atol=1e-4)
    a, b = neural.standardise(x.numpy(), x.numpy(), pooled=((0, 12), (12, 24)))
    assert np.allclose(a, b) and abs(a[:, :, :12].std() - 1.0) < 1e-3


def test_evaluation():
    import pandas as pd
    clips = pd.DataFrame({"label": [0, 0, 1, 1, 2, 2], "recording": [0, 0, 1, 1, 5, 5],
                          "participant": ["a", "a", "a", "a", "b", "b"], "fold": [0, 0, 0, 0, 1, 1]})
    proba = np.zeros((2, 6, config.N_CLASSES), dtype=np.float32)
    proba[:, [0, 1, 2, 3], [0, 0, 1, 1]] = 1.0       # the first two recordings are right
    proba[:, 4, 2], proba[:, 5, 3] = 0.6, 1.0        # the third has one right clip but the wrong mean
    summary = evaluation.summarise(proba, clips, n_boot=200)
    assert abs(summary["clip"]["mean"] - 5 / 6) < 1e-9 and summary["clip"]["std"] == 0.0
    assert abs(summary["recording"]["mean"] - 2 / 3) < 1e-9 and summary["recording"]["n"] == 3
    assert summary["clip"]["ci95"][0] <= summary["clip"]["mean"] <= summary["clip"]["ci95"][1]
    with tempfile.TemporaryDirectory() as folder:
        path = evaluation.save_predictions(folder, "toy", proba, np.ones((2, 2)), np.ones((2, 2)), {"a": 1})
        loaded = evaluation.load_predictions(path)
        assert loaded["info"] == {"a": 1} and np.array_equal(loaded["proba"].argmax(axis=2), proba.argmax(axis=2))
    results = [{"seed": s, "fold": k, "test_index": np.arange(6)[np.array(clips["fold"]) == k],
                "test": proba[0][np.array(clips["fold"]) == k], "val_accuracy": 0.5, "train_accuracy": 1.0,
                "details": {}, "n_train": 4} for s in (0, 1) for k in (0, 1)]
    joined, val, _, runs = experiments.assemble(results, 6, seeds=(0, 1), n_folds=2)
    assert np.array_equal(joined, proba) and val.shape == (2, 2) and len(runs) == 4


def test_duplicate_check_finds_a_shifted_copy():
    rng = np.random.default_rng(0)
    melody = 220.0 * 2 ** (rng.integers(-5, 6, size=(3, 200)) / 12.0)
    f0 = np.vstack([melody, melody[0] * 2 ** (3 / 12.0)])       # the last row is the first one, 3 semitones higher
    f0[:, :20] = np.nan                                          # unvoiced frames are ignored
    spread, common = features.pitch_difference_spread(f0, min_common=100)
    assert common[0, 3] == 180
    assert spread[0, 3] < 1e-6 and spread[3, 0] < 1e-6
    assert spread[0, 1] > 1.0 and spread[1, 2] > 1.0
    short, _ = features.pitch_difference_spread(f0, min_common=190)
    assert np.isinf(short[0, 1])


def test_experiment_names_are_consistent():
    for name, spec in {**experiments.CLASSICAL, **experiments.NEURAL}.items():
        assert name == f"{spec['model']}__{spec['input']}__{spec['arm']}"
    for name, variant in experiments.VARIANTS.items():
        assert variant["base"] in experiments.CLASSICAL or variant["base"] in experiments.NEURAL
        assert name.endswith(f"_{variant['variant']}__none")
    known = set(experiments.CLASSICAL) | set(experiments.NEURAL)
    for members in experiments.ENSEMBLES.values():
        assert set(members) <= known


if __name__ == "__main__":
    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print(f"{len(tests)} tests passed")
