"""Features of one 10 s clip and the model inputs built from them."""

import librosa
import numpy as np

from . import config

PITCH_FMIN = librosa.note_to_hz("C2")   # 65 Hz
PITCH_FMAX = librosa.note_to_hz("C7")   # 2093 Hz
PITCH_FMAX_WIDE = librosa.note_to_hz("C8")   # 4186 Hz, a second setting that covers high whistles
CONTOUR_STEP = 0.5                      # semitones, a smaller frame-to-frame change counts as "same pitch"
_BIGRAMS = ("++", "--", "+-", "-+", "+0", "-0", "0+", "0-", "00")
CONTOUR_STATISTIC_NAMES = (
    ("up_ratio", "down_ratio", "same_ratio", "pitch_range", "pitch_mean", "pitch_std")
    + tuple(f"pattern_{b}" for b in _BIGRAMS)
    + ("small_intervals", "medium_intervals", "large_intervals", "direction_changes")
)


def _fit_frames(rows, n_frames):
    """Pad with zeros or cut along the time axis (the last one) to exactly `n_frames`."""
    if rows.shape[-1] < n_frames:
        pad = [(0, 0)] * (rows.ndim - 1) + [(0, n_frames - rows.shape[-1])]
        return np.pad(rows, pad)
    return rows[..., :n_frames]


def frame_features(clip, sr=config.SAMPLE_RATE, n_frames=config.N_FRAMES):
    """The 54 frame-level features of one clip, shape (n_frames, 54).

    Rows in the order of `config.FEATURE_GROUPS`: MFCC (timbre), chroma (energy per pitch class), delta
    MFCC, tonnetz (tonal centroid), spectral centroid, bandwidth, roll-off and contrast, zero-crossing
    rate, RMS energy and onset strength. All use the librosa defaults (frame 2048, hop 512 samples).
    """
    mfcc = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=13)
    rows = [
        mfcc,
        librosa.feature.chroma_stft(y=clip, sr=sr),
        librosa.feature.delta(mfcc),
        librosa.feature.tonnetz(y=clip, sr=sr),
        librosa.feature.spectral_centroid(y=clip, sr=sr),
        librosa.feature.spectral_bandwidth(y=clip, sr=sr),
        librosa.feature.spectral_rolloff(y=clip, sr=sr),
        librosa.feature.spectral_contrast(y=clip, sr=sr, n_bands=3, fmin=100),
        librosa.feature.zero_crossing_rate(clip),
        librosa.feature.rms(y=clip),
        librosa.onset.onset_strength(y=clip, sr=sr)[np.newaxis, :],
    ]
    stacked = np.vstack([_fit_frames(r, n_frames) for r in rows])
    assert stacked.shape == (config.N_FEATURES, n_frames), stacked.shape
    return stacked.T.astype(np.float32)


def tempo(clip, sr=config.SAMPLE_RATE):
    """Tempo estimate of one clip in beats per minute (one number)."""
    bpm, _ = librosa.beat.beat_track(y=clip, sr=sr)
    return float(np.atleast_1d(bpm)[0])


def warm_up():
    """Compute every feature once in the calling process.

    librosa compiles some of its functions on first use and stores the result on disk. Calling this
    before worker processes start makes sure that the compilation happens once, in one process, and
    that the workers only read the stored result.
    """
    t = np.arange(3 * config.SAMPLE_RATE) / config.SAMPLE_RATE
    clip = (0.3 * np.sin(2 * np.pi * 220.0 * t) * (1.0 + np.sign(np.sin(2 * np.pi * 2.0 * t)))).astype(np.float32)
    frame_features(clip, n_frames=64)
    tempo(clip)
    pitch_track(clip, n_frames=64)
    librosa.effects.pitch_shift(clip, sr=config.SAMPLE_RATE, n_steps=0.5)
    librosa.effects.time_stretch(clip, rate=1.05)
    librosa.effects.trim(clip, top_db=config.TRIM_TOP_DB)


def aggregate(frames, tempos):
    """Mean and standard deviation of every feature over time, plus the tempo: (n, 54 + 54 + 1)."""
    return np.hstack([frames.mean(axis=1), frames.std(axis=1), np.asarray(tempos)[:, None]]).astype(np.float32)


def temporal_statistics(frames, tempos, n_parts):
    """Mean and standard deviation of every feature in each of `n_parts` consecutive parts of the clip.

    Unlike `aggregate` this keeps a coarse order in time. Shape (n, n_parts * 54 * 2 + 1).
    """
    n, n_frames, _ = frames.shape
    part = n_frames // n_parts
    columns = []
    for k in range(n_parts):
        stop = (k + 1) * part if k < n_parts - 1 else n_frames
        piece = frames[:, k * part:stop, :]
        columns += [piece.mean(axis=1), piece.std(axis=1)]
    columns.append(np.asarray(tempos)[:, None])
    return np.hstack(columns).astype(np.float32)


SEQUENCE_MODES = ("all", "melody", "melody_full")


def sequence_input(frames, tempos, mode):
    """Input of the sequence models, shape (n, n_frames, channels).

    all          the 54 frame-level features
    melody       chroma (12), onset strength (1) and the tempo repeated in every frame (1)
    melody_full  melody plus the frame-to-frame difference of the chroma (12)
    """
    if mode == "all":
        return frames.astype(np.float32)
    chroma = frames[:, :, config.feature_slice("chroma")]
    onset = frames[:, :, config.feature_slice("onset_strength")]
    tempo_channel = np.repeat(np.asarray(tempos, dtype=np.float32)[:, None, None], frames.shape[1], axis=1)
    if mode == "melody":
        parts = [chroma, onset, tempo_channel]
    elif mode == "melody_full":
        delta = np.diff(chroma, axis=1, prepend=chroma[:, :1, :])
        parts = [chroma, delta, onset, tempo_channel]
    else:
        raise ValueError(mode)
    return np.concatenate(parts, axis=2).astype(np.float32)


def pitch_track(clip, sr=config.SAMPLE_RATE, n_frames=config.N_FRAMES, fmax=PITCH_FMAX):
    """Fundamental frequency per frame in Hz from probabilistic YIN, NaN where no pitch is detected."""
    f0, _, _ = librosa.pyin(clip, fmin=PITCH_FMIN, fmax=fmax, sr=sr)
    f0 = np.where(np.isfinite(f0), f0, np.nan)[:n_frames]
    if len(f0) < n_frames:
        f0 = np.pad(f0, (0, n_frames - len(f0)), constant_values=np.nan)
    return f0.astype(np.float32)


def _filled_semitones(f0, reference):
    """Pitch in semitones relative to `reference` Hz, with unvoiced frames filled from their neighbours."""
    voiced = np.isfinite(f0) & (f0 > 0)
    if not voiced.any():
        return np.zeros(len(f0), dtype=np.float32), voiced
    index = np.arange(len(f0))
    filled = np.interp(index, index[voiced], f0[voiced])
    return (12.0 * np.log2(filled / reference)).astype(np.float32), voiced


def contour_sequence(f0):
    """Key-independent description of the melody, shape (n_frames, 3).

    Channels: pitch in semitones relative to the median pitch of the clip (0 in unvoiced frames), the
    change of the pitch from the previous frame in semitones, and a flag for voiced frames. Singing the
    same melody higher or lower leaves all three unchanged.
    """
    voiced = np.isfinite(f0) & (f0 > 0)
    if not voiced.any():
        return np.zeros((len(f0), 3), dtype=np.float32)
    semitones, _ = _filled_semitones(f0, np.median(f0[voiced]))
    interval = np.diff(semitones, prepend=semitones[:1])
    relative = np.where(voiced, semitones, 0.0)
    return np.stack([relative, interval, voiced.astype(np.float32)], axis=1).astype(np.float32)


def contour_statistics(f0):
    """Nineteen summary numbers of the pitch contour of one clip (names in CONTOUR_STATISTIC_NAMES).

    The pitch is measured in semitones relative to the first voiced frame. Every frame-to-frame change is
    labelled up, down or same (threshold CONTOUR_STEP). The statistics are the shares of the three labels,
    the range, mean and standard deviation of the pitch, the shares of the nine pairs of consecutive
    labels, the shares of small (under 2 semitones), medium and large (5 or more) changes, and the share
    of frames where the label changes.
    """
    voiced = np.isfinite(f0) & (f0 > 0)
    if voiced.sum() < 2:
        return np.zeros(len(CONTOUR_STATISTIC_NAMES), dtype=np.float32)
    semitones, _ = _filled_semitones(f0, f0[voiced][0])
    change = np.diff(semitones)
    label = np.where(change > CONTOUR_STEP, "+", np.where(change < -CONTOUR_STEP, "-", "0"))
    pairs = np.char.add(label[:-1], label[1:])
    size = np.abs(change)
    values = [
        np.mean(label == "+"), np.mean(label == "-"), np.mean(label == "0"),
        semitones.max() - semitones.min(), semitones.mean(), semitones.std(),
    ]
    values += [np.mean(pairs == b) for b in _BIGRAMS]
    values += [np.mean(size < 2), np.mean((size >= 2) & (size < 5)), np.mean(size >= 5),
               np.mean(label[:-1] != label[1:])]
    return np.asarray(values, dtype=np.float32)


def pitch_difference_spread(f0, min_common=150):
    """Standard deviation of the pitch difference (semitones) for every pair of rows, inf if the pair has
    fewer than `min_common` frames in which both are voiced. Also returns the number of such frames."""
    voiced = np.isfinite(f0) & (f0 > 0)
    semitones = np.where(voiced, 12 * np.log2(np.where(voiced, f0, 1.0)), 0.0).astype(np.float64)
    flag = voiced.astype(np.float64)
    common = flag @ flag.T
    total = semitones @ flag.T
    squares = (semitones * semitones) @ flag.T
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = (squares + squares.T - 2 * semitones @ semitones.T) / common - ((total - total.T) / common) ** 2
    spread = np.sqrt(np.maximum(variance, 0))
    spread[~(common >= min_common)] = np.inf
    return spread, common
