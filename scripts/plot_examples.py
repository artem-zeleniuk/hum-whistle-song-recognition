"""Draw a waveform, a mel spectrogram, the MFCC and a chromagram for three example recordings.

    python scripts/plot_examples.py --data-dir data/audio

The examples are the first hummed and the first whistled recording of "Happy" and the first hummed
recording of "Friend" (first in the alphabetical order of the file names). They show how the same song
looks when it is hummed and when it is whistled, and how two songs differ. Writes three PNG files to
figures/. Needs the audio, no GPU, and takes a few seconds.
"""

import argparse
import os

import librosa
import librosa.display
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import ROOT, log  # noqa: E402

from humrec import config, data  # noqa: E402

EXAMPLES = (("Happy", "hum"), ("Happy", "whistle"), ("Friend", "hum"))


def draw(path, title, out_path):
    """Four views of one recording: waveform, mel spectrogram, MFCC and chromagram."""
    y, sr = librosa.load(path, sr=config.SAMPLE_RATE)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(title, fontsize=14)
    librosa.display.waveshow(y, sr=sr, ax=axes[0, 0])
    axes[0, 0].set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
    image = librosa.display.specshow(mel, sr=sr, x_axis="time", y_axis="mel", ax=axes[0, 1])
    axes[0, 1].set_title("Mel Spectrogram")
    fig.colorbar(image, ax=axes[0, 1], format="%+2.0f dB")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    image = librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=axes[1, 0])
    axes[1, 0].set(title="MFCC (13 coefficients)", ylabel="MFCC Coefficient")
    fig.colorbar(image, ax=axes[1, 0])
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    image = librosa.display.specshow(chroma, sr=sr, x_axis="time", y_axis="chroma", ax=axes[1, 1])
    axes[1, 1].set_title("Chromagram (12 pitch classes)")
    fig.colorbar(image, ax=axes[1, 1])
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="folder with the WAV files")
    parser.add_argument("--figures-dir", default=os.path.join(ROOT, "figures"))
    args = parser.parse_args()
    os.makedirs(args.figures_dir, exist_ok=True)
    recordings = data.list_recordings(args.data_dir).sort_values("file")
    for song, kind in EXAMPLES:
        row = recordings[(recordings["song"] == song) & (recordings["kind"] == kind)].iloc[0]
        title = f"{song}, {'hummed' if kind == 'hum' else 'whistled'} ({row.file})"
        out_path = os.path.join(args.figures_dir, f"example_{song}_{kind}.png")
        draw(row.path, title, out_path)
        log(f"{row.file} -> {os.path.relpath(out_path, ROOT)}")


if __name__ == "__main__":
    main()
