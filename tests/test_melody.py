"""Tests of the melody matching on synthetic pitch tracks. They need no audio and run in a few seconds.

    python tests/test_melody.py        or        python -m pytest tests
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from humrec import config, melody  # noqa: E402


def notes(seed, n=64):
    """A random song of about half a minute: the pitch of every note in semitones and its length in seconds."""
    rng = np.random.default_rng(seed)
    return rng.integers(-7, 8, size=n).astype(np.float64), rng.choice([0.25, 0.5, 0.75], size=n)


def track(song, start=0.0, seconds=None, base=220.0, rate=1.0, noise=0.0, rests=0, seed=0):
    """Pitch track of one rendition of a song as a tracker returns it: Hz per frame, NaN in the rests.

    The rendition starts `start` seconds into the song and lasts `seconds` (to the end of the song if
    None). `base` is the frequency of pitch 0, `rate` the tempo relative to the written one, `noise`
    the standard deviation of the pitch error in semitones, `rests` the number of pauses of 0.2 s.
    """
    pitch, length = song
    ends = np.cumsum(length)
    seconds = (ends[-1] - start) / rate if seconds is None else seconds
    n_frames = int(seconds * melody.FRAME_RATE)
    place = start + np.arange(n_frames) * rate / melody.FRAME_RATE      # seconds into the song at every frame
    place = place[place < ends[-1]]
    rng = np.random.default_rng(seed)
    semitones = pitch[np.searchsorted(ends, place, side="right")] + noise * rng.standard_normal(len(place))
    f0 = base * 2.0 ** (semitones / 12.0)
    for at in rng.integers(0, len(f0) - 9, size=rests):
        f0[at:at + 9] = np.nan
    return f0.astype(np.float32)


def test_contour():
    song = notes(0)
    f0 = track(song, base=110.0)
    low, high = melody.contour(f0), melody.contour(track(song, base=392.0))
    assert low.dtype == np.float32 and len(low) == len(f0) // melody.DOWNSAMPLE
    assert abs(len(low) / (len(f0) / melody.FRAME_RATE) - 10.8) < 0.1         # values per second
    assert np.allclose(low, high, atol=1e-3)                                  # the key is gone
    assert abs(np.median(low)) < 0.5 and np.ptp(low) < 14.01                  # centred, the notes span 14 semitones
    paused = track(song, rests=10, seed=1)
    assert len(melody.contour(paused)) == np.isfinite(paused).sum() // melody.DOWNSAMPLE   # rests are removed

    steady = np.r_[np.full(140, 220.0), np.full(100, 330.0)]                  # two notes a fifth apart
    clean = melody.contour(steady)
    assert len(clean) == 60 and np.allclose(clean[:35], 0.0, atol=1e-6) and np.allclose(clean[35:], 7.02, atol=0.01)
    jump = steady.copy()
    jump[50:52] *= 2.0                                                        # an octave error of two frames
    assert np.allclose(melody.contour(jump), clean, atol=1e-6)
    lost = steady.copy()
    lost[160:180] = 4000.0                                                    # the tracker follows something else
    assert np.allclose(melody.contour(lost), clean[:-5], atol=1e-6)

    assert len(melody.contour(np.full(config.N_FRAMES, np.nan))) == 0
    short = np.full(config.N_FRAMES, np.nan)
    short[:30] = 220.0                                                        # 0.7 s of voiced sound
    assert len(melody.contour(short)) == 0


def test_excerpt_is_nearer_to_its_song_than_to_another():
    song = notes(1)
    reference = melody.contour(track(song))
    others = [melody.contour(track(notes(seed))) for seed in range(2, 8)]
    for rate in (0.8, 1.25):
        # ten seconds from the middle of the song, in another key, at another tempo, with pitch noise and rests
        query = melody.contour(track(song, start=9.0, seconds=10.0, base=311.0, rate=rate, noise=0.3, rests=5, seed=3))
        same = melody.melody_distance(query, reference)
        different = min(melody.melody_distance(query, other) for other in others)
        assert same < 0.6 and different > 2 * same, (rate, same, different)
        # an excerpt has its own median pitch: 3 semitones off costs nothing with the key search, a lot without
        assert abs(melody.melody_distance(query + 3.0, reference) - same) < 1e-4
        assert melody.melody_distance(query + 3.0, reference, transpositions=(0.0,)) > 3 * same
        assert melody.melody_distance(query + 2.75, reference) < same + 0.25 + 1e-4


def test_alignment_limits():
    song = notes(1)
    reference = melody.contour(track(song))
    cost = {rate: melody.melody_distance(melody.contour(track(song, start=5.0, seconds=8.0, rate=rate)), reference)
            for rate in (0.4, 0.5, 1.0, 2.0, 3.0)}
    assert max(cost[0.5], cost[1.0], cost[2.0]) < 0.4, cost     # half and twice the tempo are within reach
    assert min(cost[0.4], cost[3.0]) > 1.0, cost                # slower or faster renditions are not
    part = reference[100:160]
    assert melody.melody_distance(part, reference) < 1e-6       # an exact part of the reference, free start and end
    assert melody.melody_distance(part, part[:29]) == np.inf    # a reference shorter than half the query
    assert np.isfinite(melody.melody_distance(part, part[:30]))
    assert melody.subsequence_dtw(np.array([1.0, 2.0]), np.array([5.0, 0.0, 3.0, 9.0]), 1.0) == 0.5


def test_distance_matrix():
    songs = [notes(seed) for seed in range(10, 14)]
    empty = np.zeros(0, dtype=np.float32)
    references = [melody.contour(track(song)) for song in songs] + [empty]
    queries = [melody.contour(track(song, start=6.0, seconds=10.0, base=180.0, rate=1.1, noise=0.2, seed=7))
               for song in songs] + [empty]
    matrix = melody.distance_matrix(queries, references)
    assert matrix.shape == (5, 5) and matrix.dtype == np.float32
    assert np.isinf(matrix[4]).all() and np.isinf(matrix[:, 4]).all() and np.isfinite(matrix[:4, :4]).all()
    assert np.array_equal(matrix[:4, :4].argmin(axis=1), np.arange(4))
    assert np.array_equal(matrix, melody.distance_matrix(queries, references, n_jobs=2))
    assert abs(matrix[1, 2] - melody.melody_distance(queries[1], references[2])) < 1e-6
    assert (melody.distance_matrix(queries, references, transpositions=(0.0,)) >= matrix).all()
    assert melody.distance_matrix([], references).shape == (0, 5)


def test_vote():
    labels = np.array([0, 0, 1, 1, 2, 7])
    distances = np.array([0.9, 0.5, 0.4, 2.0, np.inf, 0.6])
    for k in (1, 3, 5, 40):
        for temperature in (0.01, 0.3, 10.0):
            proba = melody.vote(distances, labels, k, temperature)
            assert proba.shape == (config.N_CLASSES,) and abs(proba.sum() - 1.0) < 1e-9 and (proba >= 0).all()
    assert melody.vote(distances, labels, 1, 0.3)[1] == 1.0              # the nearest recording alone
    assert melody.vote(distances, labels, 5, 10.0).argmax() == 0         # two fairly near ones outvote it
    assert melody.vote(distances, labels, 5, 0.01).argmax() == 1         # unless the temperature is low
    assert melody.vote(distances, labels, 40, 10.0)[2] == 0.0            # no alignment, no vote
    assert abs(melody.vote(1e6 * distances, labels, 3, 0.03).sum() - 1.0) < 1e-9
    assert np.allclose(melody.vote(np.full(6, np.inf), labels, 3, 0.3), config.CHANCE_LEVEL)
    pair = np.array([[0.1, 0.9], [0.8, 0.2]])                            # two queries, two references
    proba = melody.predict(pair, [0, 1], 2, 0.3)
    assert proba.shape == (2, config.N_CLASSES) and np.array_equal(proba.argmax(axis=1), [0, 1])
    k, temperature, table = melody.choose_settings(pair, [0, 1], [0, 1], k_grid=(1, 2), temperature_grid=(0.03, 1.0))
    assert (table == 1.0).all() and (k, temperature) == (1, 0.03)        # equal accuracies go to the first pair


def test_songs_are_recognised():
    songs = [notes(100 + label) for label in range(config.N_CLASSES)]

    def rendition(label, seed, excerpt):
        rng = np.random.default_rng(seed)
        start, seconds = (rng.uniform(0.0, 12.0), 10.0) if excerpt else (0.0, None)
        return melody.contour(track(songs[label], start=start, seconds=seconds, base=rng.uniform(100.0, 400.0),
                                    rate=rng.uniform(0.8, 1.25), noise=0.3, rests=4, seed=seed))

    labels = np.repeat(np.arange(config.N_CLASSES), 3)                   # three whole renditions of every song
    references = [rendition(label, 1000 + index, False) for index, label in enumerate(labels)]
    y = np.repeat(np.arange(config.N_CLASSES), 2)                        # two excerpts of every song per set
    validation = [rendition(label, 2000 + index, True) for index, label in enumerate(y)]
    test = [rendition(label, 3000 + index, True) for index, label in enumerate(y)]
    test.append(np.zeros(0, dtype=np.float32))                           # a clip without a pitch
    distances = melody.distance_matrix(validation, references, n_jobs=2)
    k_grid, temperature_grid = (1, 3, 24), (0.03, 10.0)
    k, temperature, table = melody.choose_settings(distances, y, labels, k_grid, temperature_grid)
    assert table.shape == (3, 2) and table.max() > 0.9
    assert table[0, 0] == table[0, 1]                                    # one voter, the temperature has no say
    assert table[k_grid.index(k), temperature_grid.index(temperature)] == table.max()
    proba = melody.predict(melody.distance_matrix(test, references, n_jobs=2), labels, k, temperature)
    assert proba.shape == (17, config.N_CLASSES) and np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert np.mean(proba[:16].argmax(axis=1) == y) > 0.9
    assert np.allclose(proba[16], config.CHANCE_LEVEL)


if __name__ == "__main__":
    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print(f"{len(tests)} tests passed")
