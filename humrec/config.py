"""Constants shared by the library, the scripts and the tests."""

# The eight songs of the data set, in the order of the class index.
SONGS = (
    "Feeling",
    "Friend",
    "Happy",
    "Married",
    "Necessities",
    "NewYork",
    "RememberMe",
    "TryEverything",
)
N_CLASSES = len(SONGS)
CHANCE_LEVEL = 1.0 / N_CLASSES

# Audio.
SAMPLE_RATE = 22050        # every recording is resampled to this rate (Hz) and mixed down to one channel
CLIP_SECONDS = 10.0        # length of one model input
STEP_SECONDS = 5.0         # distance between the starts of consecutive clips of one recording
MIN_SECONDS = 5.0          # shorter leftovers are dropped, longer ones are padded with zeros to a full clip
TRIM_TOP_DB = 20           # leading and trailing audio quieter than this (dB below the peak) is cut
CLIP_SAMPLES = int(CLIP_SECONDS * SAMPLE_RATE)

# Frame-level features.
HOP_LENGTH = 512           # librosa default, 23 ms at 22050 Hz
N_FRAMES = 431             # frames of one 10 s clip
FEATURE_GROUPS = (         # (name, number of rows), in the order of the feature axis
    ("mfcc", 13),
    ("chroma", 12),
    ("delta_mfcc", 13),
    ("tonnetz", 6),
    ("spectral_centroid", 1),
    ("spectral_bandwidth", 1),
    ("spectral_rolloff", 1),
    ("spectral_contrast", 4),
    ("zcr", 1),
    ("rms", 1),
    ("onset_strength", 1),
)
N_FEATURES = sum(n for _, n in FEATURE_GROUPS)  # 54


def feature_names():
    """Names of the 54 frame-level features, in the order of the feature axis."""
    names = []
    for group, n in FEATURE_GROUPS:
        names += [group] if n == 1 else [f"{group}_{i + 1}" for i in range(n)]
    return names


def feature_slice(group):
    """Index range of one feature group on the feature axis."""
    start = 0
    for name, n in FEATURE_GROUPS:
        if name == group:
            return slice(start, start + n)
        start += n
    raise KeyError(group)


# Evaluation protocol.
N_FOLDS = 5                # participant-disjoint folds, every recording is tested exactly once
FOLD_SEED = 0              # seed of the division into folds
SEEDS = (0, 1, 2, 3, 4)    # training seeds of every model that has a random component

# Augmentation of training clips. Every transform of an arm is applied to a clip with this probability,
# independently of the others, and each application adds one more training clip.
AUGMENT_PROBABILITY = 0.4
AUGMENTATION_ARMS = {
    # small changes meant to leave the melody intact
    "conservative": (("noise", 1e-4), ("pitch", 0.5)),
    # the ranges that are common defaults for speech and music
    "strong": (("noise", 5e-3), ("pitch", 2.0), ("stretch", 0.10)),
}
# A third arm of another kind: every training clip gets this many extra copies, and each copy is changed by
# all three transforms at once (a random tempo, then a random pitch shift, then noise).
COMBINED_ARMS = {
    "heavy": {"copies": 4, "stretch": 0.15, "pitch": 3.0, "noise": 5e-3},
}
AUGMENT_SEED = 0

# Public source of the audio (a 1.5 GB zip archive with 800 WAV files).
DATA_URL = "https://github.com/thekmannn/MLEndHW_Sample/raw/main/MLEndHWII_Sample_800.zip"
