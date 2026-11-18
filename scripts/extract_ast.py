"""Embed every clip with a pretrained Audio Spectrogram Transformer (weights stay frozen).

    python scripts/extract_ast.py --data-dir data/audio

Each clip is resampled to 16 kHz and turned into a log-mel spectrogram by the feature extractor of the
model. The embedding is the output at the classification token of the last layer (768 numbers).
Writes ast_cls.npy (clips, 768) to the cache folder.
"""

import argparse
import os

import librosa
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from _common import add_common_arguments, log

from humrec import audio, config, data

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_RATE = 16000


def spectrograms(path, expected, extractor):
    """Model inputs (clips, 1024, 128) of one recording."""
    clips = audio.split_into_clips(audio.load_audio(path))
    if len(clips) != expected:
        raise RuntimeError(f"{path}: {len(clips)} clips, expected {expected}")
    waves = [librosa.resample(clip, orig_sr=config.SAMPLE_RATE, target_sr=AST_RATE) for _, clip, _ in clips]
    return extractor(waves, sampling_rate=AST_RATE, return_tensors="np")["input_values"].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_common_arguments(parser)
    args = parser.parse_args()
    from transformers import ASTFeatureExtractor, ASTModel

    clips = pd.read_csv(os.path.join(args.cache_dir, "clips.csv"))
    counts = clips.groupby("file", sort=False).size()
    paths = {os.path.basename(p): p for p in data.find_wav_files(args.data_dir)}
    extractor = ASTFeatureExtractor.from_pretrained(MODEL_NAME)
    model = ASTModel.from_pretrained(MODEL_NAME).to(args.device).eval()
    log(f"{len(clips)} clips, model {MODEL_NAME} on {args.device}")

    files = list(counts.index)
    embeddings = []
    for start in range(0, len(files), 100):
        batch = files[start:start + 100]
        inputs = np.concatenate(Parallel(n_jobs=min(args.workers, 16))(
            delayed(spectrograms)(paths[f], int(counts[f]), extractor) for f in batch))
        with torch.no_grad():
            for i in range(0, len(inputs), args.batch_size):
                values = torch.as_tensor(inputs[i:i + args.batch_size], device=args.device)
                embeddings.append(model(values).last_hidden_state[:, 0, :].float().cpu().numpy())
        log(f"{min(start + 100, len(files))} of {len(files)} recordings embedded")
    embeddings = np.concatenate(embeddings)
    assert len(embeddings) == len(clips), (len(embeddings), len(clips))
    np.save(os.path.join(args.cache_dir, "ast_cls.npy"), embeddings.astype(np.float32))
    log(f"ast_cls.npy written, shape {embeddings.shape}")


if __name__ == "__main__":
    main()
