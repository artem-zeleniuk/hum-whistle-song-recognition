"""Pretrained audio encoders: one embedding per layer for every clip, and the list of encoders."""

import numpy as np
import torch

from . import config

# name -> Hugging Face model, input sample rate and family (how the input is prepared and pooled).
ENCODERS = {
    "ast": {"model": "MIT/ast-finetuned-audioset-10-10-0.4593", "rate": 16000, "family": "ast"},
    "wav2vec2": {"model": "facebook/wav2vec2-base", "rate": 16000, "family": "wave"},
    "hubert": {"model": "facebook/hubert-base-ls960", "rate": 16000, "family": "wave"},
    "wavlm": {"model": "microsoft/wavlm-base-plus", "rate": 16000, "family": "wave"},
    "whisper": {"model": "openai/whisper-small", "rate": 16000, "family": "whisper"},
    # MERT is a music encoder with a HuBERT layout. Its model class lives in the model repository, so the
    # revision is pinned and the code of that revision is what runs (licence of the weights: CC BY-NC 4.0).
    "mert": {"model": "m-a-p/MERT-v1-95M", "rate": 24000, "family": "wave", "remote_code": True,
             "revision": "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"},
    "mert330": {"model": "m-a-p/MERT-v1-330M", "rate": 24000, "family": "wave", "remote_code": True,
                "revision": "5240c2708a5acaee1007f43fb9735c7dcd0b78c9"},
}
WHISPER_FRAMES_PER_SECOND = 50


def to_int16(clip):
    """Store a float waveform in [-1, 1] as 16-bit integers (the precision of the source files)."""
    return np.clip(np.round(clip * 32767.0), -32768, 32767).astype(np.int16)


def resample(waves, rate, device):
    """int16 clips at the working sample rate -> float32 tensor at `rate` Hz on `device`."""
    import librosa
    x = np.asarray(waves, dtype=np.float32) / 32767.0
    if rate != config.SAMPLE_RATE:
        x = librosa.resample(x, orig_sr=config.SAMPLE_RATE, target_sr=rate, axis=-1)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def load_model(name):
    """The pretrained model of one encoder (for Whisper only its encoder half)."""
    from transformers import AutoConfig, AutoModel
    spec = ENCODERS[name]
    if spec.get("remote_code"):
        settings = AutoConfig.from_pretrained(spec["model"], revision=spec["revision"], trust_remote_code=True)
        # an option that newer versions of transformers expect in every HuBERT-like configuration
        if not hasattr(settings, "conv_pos_batch_norm"):
            settings.conv_pos_batch_norm = False
        return AutoModel.from_pretrained(spec["model"], config=settings, revision=spec["revision"],
                                         trust_remote_code=True)
    model = AutoModel.from_pretrained(spec["model"])
    return model.encoder if spec["family"] == "whisper" else model


def load_processor(name):
    from transformers import AutoFeatureExtractor
    spec = ENCODERS[name]
    return AutoFeatureExtractor.from_pretrained(spec["model"], revision=spec.get("revision"))


def load_encoder(name, device):
    """The model and its input processor, in evaluation mode with frozen weights."""
    processor = load_processor(name)
    model = load_model(name)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, processor


def _pool(states, real_frames):
    """Mean over time of every layer, using only the frames that hold real audio. Returns (batch, layers, dim)."""
    stacked = torch.stack(states, dim=1).float()                      # (batch, layers, frames, dim)
    frames = stacked.shape[2]
    index = torch.arange(frames, device=stacked.device)[None, :]
    mask = (index < real_frames[:, None]).float()[:, None, :, None]   # (batch, 1, frames, 1)
    return (stacked * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)


@torch.no_grad()
def embed(name, model, processor, waves, real_seconds, device):
    """Layer-wise embeddings of a batch of clips.

    waves         (batch, samples) int16 at config.SAMPLE_RATE
    real_seconds  (batch,) seconds of real audio in each clip, the zero-padded rest is left out of the mean
    Returns a float32 array (batch, layers, dim): for every layer of the encoder (the input embedding
    first) the mean of its output over time. For the spectrogram transformer the mean runs over all
    patches, which cannot be separated into real and padded time.
    """
    spec = ENCODERS[name]
    audio = resample(waves, spec["rate"], device)
    real = torch.as_tensor(np.asarray(real_seconds), dtype=torch.float32, device=device)
    if spec["family"] == "wave":
        if getattr(processor, "do_normalize", False):
            audio = (audio - audio.mean(dim=1, keepdim=True)) / (audio.std(dim=1, keepdim=True) + 1e-7)
        states = model(audio, output_hidden_states=True).hidden_states
        frames = states[0].shape[1]
        return _pool(states, torch.ceil(real / config.CLIP_SECONDS * frames)).cpu().numpy()
    if spec["family"] == "whisper":
        features = processor(list(audio.cpu().numpy()), sampling_rate=spec["rate"], return_tensors="pt")
        states = model(features["input_features"].to(device), output_hidden_states=True).hidden_states
        return _pool(states, torch.ceil(real * WHISPER_FRAMES_PER_SECOND)).cpu().numpy()
    if spec["family"] == "ast":
        features = processor(list(audio.cpu().numpy()), sampling_rate=spec["rate"], return_tensors="pt")
        states = model(features["input_values"].to(device), output_hidden_states=True).hidden_states
        patches = torch.stack(states, dim=1)[:, :, 2:, :].float()      # drop the two special tokens
        return patches.mean(dim=2).cpu().numpy()
    raise ValueError(spec["family"])
