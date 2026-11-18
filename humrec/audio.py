"""Loading a recording, cutting it into 10 s clips and augmenting a clip."""

import zlib

import librosa
import numpy as np

from . import config


def load_audio(path, sr=config.SAMPLE_RATE):
    """Read a WAV file as one channel of float32 samples at `sr` Hz."""
    samples, _ = librosa.load(path, sr=sr, mono=True)
    return samples


def split_into_clips(samples, sr=config.SAMPLE_RATE, clip_seconds=config.CLIP_SECONDS,
                     step_seconds=config.STEP_SECONDS, min_seconds=config.MIN_SECONDS,
                     trim_top_db=config.TRIM_TOP_DB):
    """Cut one recording into overlapping clips of equal length.

    Silence at both ends is removed first. A window of `clip_seconds` then moves over the rest in steps of
    `step_seconds`. What is left after the last full window is kept if it is at least `min_seconds` long
    and is padded with zeros to a full clip. A recording shorter than `min_seconds` after trimming gives
    no clip.

    Returns a list of (start in seconds from the start of the file, clip, seconds of real audio in the clip).
    """
    offset = 0
    if trim_top_db is not None:
        samples, (offset, _) = librosa.effects.trim(samples, top_db=trim_top_db)
    if len(samples) < min_seconds * sr:
        return []
    clip_len = int(clip_seconds * sr)
    step = int(step_seconds * sr)
    clips = []
    start = 0
    while start + clip_len <= len(samples):
        clips.append(((offset + start) / sr, samples[start:start + clip_len].astype(np.float32), clip_seconds))
        start += step
    rest = len(samples) - start
    if not clips or rest >= min_seconds * sr:
        tail = samples[start:]
        padded = np.pad(tail, (0, clip_len - len(tail))).astype(np.float32)
        clips.append(((offset + start) / sr, padded, len(tail) / sr))
    return clips


def add_noise(clip, amplitude, rng):
    """Add white Gaussian noise with the given standard deviation (full scale is 1)."""
    return (clip + amplitude * rng.standard_normal(len(clip))).astype(np.float32)


def shift_pitch(clip, semitones, sr=config.SAMPLE_RATE):
    """Move every pitch by `semitones` without changing the length."""
    return librosa.effects.pitch_shift(clip, sr=sr, n_steps=semitones).astype(np.float32)


def stretch_time(clip, rate):
    """Play `rate` times faster without changing the pitch, then cut or pad to the original length."""
    out = librosa.effects.time_stretch(clip, rate=rate)
    if len(out) < len(clip):
        out = np.pad(out, (0, len(clip) - len(out)))
    return out[:len(clip)].astype(np.float32)


def clip_rng(file_name, clip_index, arm):
    """Random generator of one clip and one augmentation arm.

    It depends only on the file name, the clip number and the arm, so the augmented training set is the
    same in every run and does not depend on the number of worker processes.
    """
    return np.random.default_rng(
        [config.AUGMENT_SEED, zlib.crc32(file_name.encode()), clip_index, zlib.crc32(arm.encode())])


def augmented_versions(clip, arm, rng, sr=config.SAMPLE_RATE):
    """Augmented copies of one clip for one arm of `config.AUGMENTATION_ARMS` or `config.COMBINED_ARMS`.

    In an arm of the first kind every transform is applied to the original clip with probability
    `config.AUGMENT_PROBABILITY`, independently of the others, and each application gives one copy. In a
    combined arm every clip gets a fixed number of copies, each changed by all transforms at once.
    Returns a list of (transform name, clip), which can be empty for an arm of the first kind.
    """
    out = []
    if arm in config.COMBINED_ARMS:
        spec = config.COMBINED_ARMS[arm]
        for _ in range(spec["copies"]):
            changed = stretch_time(clip, rng.uniform(1.0 - spec["stretch"], 1.0 + spec["stretch"]))
            changed = shift_pitch(changed, rng.uniform(-spec["pitch"], spec["pitch"]), sr)
            out.append(("combined", add_noise(changed, spec["noise"], rng)))
        return out
    for kind, size in config.AUGMENTATION_ARMS[arm]:
        if rng.random() >= config.AUGMENT_PROBABILITY:
            continue
        if kind == "noise":
            out.append((kind, add_noise(clip, size, rng)))
        elif kind == "pitch":
            out.append((kind, shift_pitch(clip, rng.uniform(-size, size), sr)))
        elif kind == "stretch":
            out.append((kind, stretch_time(clip, rng.uniform(1.0 - size, 1.0 + size))))
        else:
            raise ValueError(kind)
    return out
