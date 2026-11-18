"""Melody matching: pitch contours compared by dynamic time warping, and a vote of the nearest recordings.

A hummed melody is defined by its pitch contour up to the key (all pitches moved by the same interval)
and the tempo. The contour of a query clip is compared with the contour of every reference recording
under both changes: a small set of transpositions of the query is tried, and the alignment may stretch
or compress time by up to a factor of two. The query may match any contiguous part of the reference,
which is usually longer. The nearest references then vote for their song.
"""

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from numba import njit
from scipy.ndimage import median_filter

from . import config

FRAME_RATE = config.SAMPLE_RATE / config.HOP_LENGTH     # 43.07 pitch frames per second
MEDIAN_FRAMES = 5                                       # width of the median filter, 116 ms
DOWNSAMPLE = 4                                          # frames per contour value, 10.8 values per second
MAX_SEMITONES = 18.0                                    # frames further from the median pitch are tracker errors
MIN_VOICED_SECONDS = 1.0                                # a track with less voiced sound gives no contour
TRANSPOSITIONS = tuple(0.5 * s for s in range(-8, 9))   # semitones added to the query in the key search
K_GRID = (1, 3, 5, 10, 20, 40)                          # numbers of voters compared on the validation fold
TEMPERATURE_GRID = (0.03, 0.1, 0.3, 1.0)                # semitones, scale of the vote weights


def contour(f0, median_frames=MEDIAN_FRAMES, downsample=DOWNSAMPLE, max_semitones=MAX_SEMITONES,
            min_voiced_seconds=MIN_VOICED_SECONDS):
    """Compact key-normalised contour of one pitch track (Hz per frame, NaN where unvoiced).

    1. Unvoiced frames are removed, not interpolated. Rests and breaths differ from one person to the
       next, and the zero padding at the end of a short clip is unvoiced as well. Removing them shortens
       the time axis, which the time warping absorbs.
    2. Voiced frames further than `max_semitones` from the median pitch are removed too. No melody
       reaches that far, such frames come from the tracker locking on to noise or to a high harmonic.
    3. The pitch is expressed in semitones relative to the median of the remaining frames, so the same
       melody in another key gives the same contour.
    4. A median filter over `median_frames` frames removes octave jumps and other tracker errors that
       last one or two frames.
    5. Consecutive blocks of `downsample` frames are averaged.

    Returns a float32 array with about 10 values per second of voiced sound, empty if less than
    `min_voiced_seconds` of usable frames are left.
    """
    f0 = np.asarray(f0, dtype=np.float64)
    semitones = 12.0 * np.log2(f0[np.isfinite(f0) & (f0 > 0)])
    if len(semitones):
        semitones = semitones[np.abs(semitones - np.median(semitones)) <= max_semitones]
    if len(semitones) < min_voiced_seconds * FRAME_RATE:
        return np.zeros(0, dtype=np.float32)
    semitones -= np.median(semitones)
    smooth = median_filter(semitones, size=median_frames, mode="nearest")
    n = len(smooth) // downsample
    return smooth[:n * downsample].reshape(n, downsample).mean(axis=1).astype(np.float32)


@njit(nogil=True)
def subsequence_dtw(query, reference, shift):
    """Cost of the best alignment of the whole query with a contiguous part of the reference.

    Both are float64 contours in semitones, `shift` is added to every query value. The alignment may
    start and end anywhere on the reference. It is built from three steps:
      one query value on the next reference value            (same tempo)
      one query value on the reference value after the next  (the query is up to twice as fast)
      two query values on the next reference value           (the query is up to twice as slow)
    so the tempo of the query stays between half and twice the tempo of the reference, locally and as
    a whole. Every query value is matched to exactly one reference value, and the cost is the mean
    absolute difference in semitones over the query values. Returns infinity if no alignment exists
    (the reference is shorter than half the query).
    """
    n, m = len(query), len(reference)
    # rows of the cost table over the reference position, for i - 2, i - 1 and i matched query values
    before = np.full(m + 1, np.inf)
    last = np.zeros(m + 1)              # no query value matched yet: free start on the reference
    row = np.empty(m + 1)
    for i in range(1, n + 1):
        value = query[i - 1] + shift
        row[0] = np.inf
        for j in range(1, m + 1):
            cost = abs(value - reference[j - 1])
            best = last[j - 1]
            if j >= 2 and last[j - 2] < best:
                best = last[j - 2]
            best += cost
            if i >= 2:
                both = before[j - 1] + abs(query[i - 2] + shift - reference[j - 1]) + cost
                if both < best:
                    best = both
            row[j] = best
        before, last, row = last, row, before
    return last.min() / n               # free end on the reference


@njit(nogil=True)
def _distance(query, reference, transpositions):
    best = np.inf
    if len(query) == 0 or len(reference) == 0:
        return best
    for shift in transpositions:
        best = min(best, subsequence_dtw(query, reference, shift))
    return best


@njit(nogil=True)
def _distance_rows(query_values, query_starts, first, stop, reference_values, reference_starts, transpositions):
    """Distances of the queries `first` to `stop - 1` to every reference (contours packed by `_pack`)."""
    n_references = len(reference_starts) - 1
    out = np.full((stop - first, n_references), np.inf)
    for a in range(first, stop):
        query = query_values[query_starts[a]:query_starts[a + 1]]
        for b in range(n_references):
            reference = reference_values[reference_starts[b]:reference_starts[b + 1]]
            out[a - first, b] = _distance(query, reference, transpositions)
    return out


def _pack(contours):
    """Join contours of different lengths into one float64 array plus the start index of each."""
    starts = np.zeros(len(contours) + 1, dtype=np.int64)
    starts[1:] = np.cumsum([len(c) for c in contours])
    values = np.zeros(starts[-1], dtype=np.float64)
    for index, item in enumerate(contours):
        values[starts[index]:starts[index + 1]] = item
    return values, starts


def melody_distance(query, reference, transpositions=TRANSPOSITIONS):
    """Distance between a query contour and a reference contour, in semitones per query value.

    The minimum of `subsequence_dtw` over the transpositions of the query (in semitones, by default
    -4 to 4 in steps of a half). The search is needed although both contours are relative to their own
    median pitch: the median of a 10 s excerpt differs from the median of the whole melody. Returns
    infinity if one of the contours is empty or the reference is too short to hold the query.
    """
    return float(_distance(np.asarray(query, dtype=np.float64), np.asarray(reference, dtype=np.float64),
                           np.asarray(transpositions, dtype=np.float64)))


def distance_matrix(queries, references, transpositions=TRANSPOSITIONS, n_jobs=1):
    """`melody_distance` of every query to every reference, shape (queries, references), float32.

    The rows are computed in `n_jobs` threads. The compiled code releases the interpreter lock, so
    threads run in parallel and the contours are not copied to other processes.
    """
    query_values, query_starts = _pack(queries)
    reference_values, reference_starts = _pack(references)
    shifts = np.asarray(transpositions, dtype=np.float64)
    if len(queries) == 0:
        return np.zeros((0, len(references)), dtype=np.float32)
    n_parts = min(len(queries), 8 * effective_n_jobs(n_jobs))
    bounds = np.linspace(0, len(queries), n_parts + 1).astype(np.int64)
    # a call without queries, so that the code is compiled once here and not by several threads at a time
    _distance_rows(query_values, query_starts, bounds[0], bounds[0], reference_values, reference_starts, shifts)
    parts = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_distance_rows)(query_values, query_starts, bounds[p], bounds[p + 1],
                                reference_values, reference_starts, shifts) for p in range(n_parts))
    return np.concatenate(parts).astype(np.float32)


def vote(distances, labels, k, temperature, n_classes=config.N_CLASSES):
    """Class probabilities of one query from its distances to the reference recordings.

    The `k` nearest references vote for their class (`labels`) with the weight
    exp(-distance / temperature). Returns `n_classes` probabilities that sum to 1. They are all equal
    if no distance is finite, which is the case for a query with an empty contour.
    """
    distances = np.asarray(distances, dtype=np.float64)
    usable = np.flatnonzero(np.isfinite(distances))
    if len(usable) == 0:
        return np.full(n_classes, 1.0 / n_classes)
    nearest = usable[np.argsort(distances[usable], kind="stable")[:k]]
    # relative to the nearest one, which leaves the probabilities unchanged and cannot underflow to all zeros
    weights = np.exp(-(distances[nearest] - distances[nearest[0]]) / temperature)
    votes = np.bincount(np.asarray(labels)[nearest], weights=weights, minlength=n_classes)
    return votes / votes.sum()


def predict(distances, labels, k, temperature, n_classes=config.N_CLASSES):
    """`vote` for every row of a (queries, references) distance matrix, shape (queries, n_classes)."""
    out = np.zeros((len(distances), n_classes), dtype=np.float32)
    for index, row in enumerate(distances):
        out[index] = vote(row, labels, k, temperature, n_classes)
    return out


def choose_settings(distances, y, labels, k_grid=K_GRID, temperature_grid=TEMPERATURE_GRID):
    """The number of voters and the temperature with the best accuracy on one set of queries.

    `distances` is (queries, references), `y` the true class of the queries, `labels` the class of the
    references. Returns (k, temperature, table), where table[a, b] is the accuracy with k_grid[a] and
    temperature_grid[b]. Equal accuracies go to the smaller k, then to the lower temperature.
    """
    y = np.asarray(y)
    table = np.zeros((len(k_grid), len(temperature_grid)))
    for a, k in enumerate(k_grid):
        for b, temperature in enumerate(temperature_grid):
            table[a, b] = np.mean(predict(distances, labels, k, temperature).argmax(axis=1) == y)
    a, b = np.unravel_index(int(table.argmax()), table.shape)
    return int(k_grid[a]), float(temperature_grid[b]), table
