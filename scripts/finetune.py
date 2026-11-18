"""Fine-tune a pretrained audio encoder end to end on the 10 s clips.

    python scripts/finetune.py --encoder hubert
    python scripts/finetune.py --encoder hubert --shard 0 4                # every fourth run, e.g. one GPU of four

The encoder (humrec/pretrained.py) gets a linear layer on the mean of its last hidden states and all
weights are trained, except the convolutional waveform front end of the wav2vec 2.0 family, which
stays frozen. AdamW, a small learning rate for the encoder and a larger one for the new layer, linear
warm-up and decay, label smoothing, mixed precision. After every epoch the validation accuracy is
measured, and the test predictions of the best epoch are kept (with the accuracy on 512 training clips
at that epoch). Every run (fold and seed) is saved to
results/finetune_parts as soon as it ends, so an interrupted job continues where it stopped. When all
runs of an encoder exist they are joined into results/predictions/finetune__<encoder>__none.npz.
The saved runs are told apart by encoder, learning rate, fold and seed only, and the joined file by the
encoder only: use another results folder when the number of epochs or the batch size changes, and join
only the learning rate that is to be kept.
Needs waves.npy in the cache folder (written by scripts/extract_embeddings.py).
"""

import argparse
import os
import time

import numpy as np
import torch
from torch import nn

from _common import add_common_arguments, log

from humrec import config, data, evaluation, experiments, pretrained


class FineTuned(nn.Module):
    def __init__(self, name, tiny=False):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        spec = pretrained.ENCODERS[name]
        self.family = spec["family"]
        if tiny:  # a small untrained model of the same family, only to test the code without a download
            settings = AutoConfig.for_model(
                "wav2vec2", hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                conv_dim=(16, 16, 16), conv_stride=(5, 4, 4), conv_kernel=(10, 8, 8), num_feat_extract_layers=3,
                num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2)
            self.encoder = AutoModel.from_config(settings)
            self.family = "wave"
        else:
            self.encoder = pretrained.load_model(name)
        if self.family == "wave":
            # the convolutional front end that turns the waveform into frames stays as pretrained
            for parameter in self.encoder.feature_extractor.parameters():
                parameter.requires_grad_(False)
        self.head = nn.Linear(self.encoder.config.hidden_size, config.N_CLASSES)

    def forward(self, inputs, real_share):
        states = self.encoder(inputs).last_hidden_state
        if self.family == "ast":
            return self.head(states[:, 2:, :].mean(dim=1))
        frames = states.shape[1]
        index = torch.arange(frames, device=states.device)[None, :]
        mask = (index < torch.ceil(real_share * frames)[:, None]).to(states.dtype)[:, :, None]
        return self.head((states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0))


def prepare_inputs(name, waves, device, tiny=False):
    """All clips in the input format of the encoder, as one tensor on the device."""
    spec = pretrained.ENCODERS[name]
    family = "wave" if tiny else spec["family"]
    parts = []
    processor = None
    if family == "ast":
        processor = pretrained.load_processor(name)
    normalise = False
    if family == "wave" and not tiny:
        normalise = bool(getattr(pretrained.load_processor(name), "do_normalize", False))
    for start in range(0, len(waves), 64):
        audio = pretrained.resample(waves[start:start + 64], spec["rate"], device)
        if family == "ast":
            batch = processor(list(audio.cpu().numpy()), sampling_rate=spec["rate"], return_tensors="pt")
            parts.append(batch["input_values"].to(device=device, dtype=torch.float16))
        else:
            if normalise:
                audio = (audio - audio.mean(dim=1, keepdim=True)) / (audio.std(dim=1, keepdim=True) + 1e-7)
            parts.append(audio.to(torch.float16))
    return torch.cat(parts)


def run(name, inputs, real_share, labels, roles, seed, device, epochs, batch_size, lr, tiny=False):
    """One fine-tuning run. Returns a result dict in the format of experiments.assemble."""
    train, val, test = (torch.as_tensor(np.flatnonzero(mask), device=device) for mask in roles)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FineTuned(name, tiny=tiny).to(device)
    optimiser = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": lr},
                                   {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    steps = epochs * int(np.ceil(len(train) / batch_size))
    warm = max(1, int(0.1 * steps))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: (step + 1) / warm if step < warm else max(0.0, (steps - step) / max(1, steps - warm)))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    order = torch.Generator().manual_seed(seed)

    def predict(index):
        model.eval()
        out = []
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
            for start in range(0, len(index), 2 * batch_size):
                rows = index[start:start + 2 * batch_size]
                out.append(torch.softmax(model(inputs[rows].float(), real_share[rows]).float(), dim=1))
        return torch.cat(out).cpu().numpy()

    best = {"accuracy": -1.0}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = train[torch.randperm(len(train), generator=order).to(device)]
        for start in range(0, len(permutation) - 1, batch_size):
            rows = permutation[start:start + batch_size]
            optimiser.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = loss_fn(model(inputs[rows].float(), real_share[rows]).float(), labels[rows])
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
        val_proba = predict(val)
        accuracy = float(np.mean(val_proba.argmax(axis=1) == labels[val].cpu().numpy()))
        history.append(round(accuracy, 4))
        if accuracy > best["accuracy"]:
            # the training accuracy is measured on 512 training clips, which is enough to see overfitting
            sample = train[:512]
            train_accuracy = float(np.mean(predict(sample).argmax(axis=1) == labels[sample].cpu().numpy()))
            best = {"accuracy": accuracy, "epoch": epoch, "test": predict(test), "train": train_accuracy}
    return {"test": best["test"].astype(np.float32), "test_index": test.cpu().numpy(), "val_accuracy": best["accuracy"],
            "train_accuracy": best["train"], "n_train": int(len(train)),
            "details": {"best_epoch": best["epoch"], "epochs": epochs, "val_accuracy_per_epoch": history}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", required=True, choices=sorted(pretrained.ENCODERS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(config.SEEDS))
    parser.add_argument("--folds", nargs="*", type=int, default=list(range(config.N_FOLDS)))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate of the encoder")
    parser.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("INDEX", "COUNT"),
                        help="do only the runs whose number modulo COUNT is INDEX (to share the runs between GPUs)")
    parser.add_argument("--no-join", action="store_true", help="do not write the joined prediction file")
    parser.add_argument("--arm", default="none",
                        help="augmentation arm whose changed clips join the training set (needs waves_<arm>.npy and "
                             "parent_<arm>.npy in the cache, written by scripts/extract_features.py --keep-waves)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tiny-random-model", action="store_true",
                        help="use a small untrained model instead of the pretrained one (code test without a download)")
    add_common_arguments(parser)
    args = parser.parse_args()
    cache = experiments.Cache(args.cache_dir)
    name = f"finetune__{args.encoder}__{args.arm}"
    parts_dir = os.path.join(args.results_dir, "finetune_parts")
    os.makedirs(parts_dir, exist_ok=True)
    waves = np.load(os.path.join(args.cache_dir, "waves.npy"), mmap_mode="r")
    inputs = prepare_inputs(args.encoder, waves, args.device, tiny=args.tiny_random_model)
    real_share = torch.as_tensor(cache.clips["real_seconds"].to_numpy() / config.CLIP_SECONDS, dtype=torch.float32,
                                 device=args.device)
    labels = torch.as_tensor(cache.labels, dtype=torch.long, device=args.device)
    parent = None
    if args.arm != "none":
        # the changed clips are appended after the originals, each with the label and the length of its original
        parent = cache.parent(args.arm)
        changed = np.load(os.path.join(args.cache_dir, f"waves_{args.arm}.npy"), mmap_mode="r")
        added = prepare_inputs(args.encoder, changed, args.device, tiny=args.tiny_random_model)
        assert len(added) == len(parent), (len(added), len(parent))
        inputs = torch.cat([inputs, added])
        index = torch.as_tensor(parent, device=args.device)
        real_share = torch.cat([real_share, real_share[index]])
        labels = torch.cat([labels, labels[index]])
    log(f"{name}: inputs {tuple(inputs.shape)} on {args.device}, folds {args.folds}, seeds {args.seeds}")

    def part_path(fold, seed):
        return os.path.join(parts_dir, f"{name}__lr{args.lr:g}__fold{fold}_seed{seed}.npz")

    for number, (seed, fold) in enumerate((seed, fold) for seed in args.seeds for fold in args.folds):
        if number % args.shard[1] != args.shard[0]:
            continue
        part = part_path(fold, seed)
        if os.path.exists(part):
            continue
        started = time.time()
        train, val, test = data.fold_roles(cache.fold, fold)
        if parent is not None:
            # originals keep their roles, a changed clip is a training clip if its original is one
            rest = np.zeros(len(parent), dtype=bool)
            train, val, test = (np.concatenate([train, train[parent]]), np.concatenate([val, rest]),
                                np.concatenate([test, rest]))
        result = run(args.encoder, inputs, real_share, labels, (train, val, test), seed, args.device,
                     args.epochs, args.batch_size, args.lr, tiny=args.tiny_random_model)
        np.savez_compressed(part + ".tmp.npz", test=result["test"], test_index=result["test_index"],
                            val_accuracy=result["val_accuracy"], train_accuracy=result["train_accuracy"],
                            n_train=result["n_train"], best_epoch=result["details"]["best_epoch"],
                            history=np.asarray(result["details"]["val_accuracy_per_epoch"]))
        os.replace(part + ".tmp.npz", part)
        log(f"{name} fold {fold} seed {seed}: validation {result['val_accuracy']:.4f} at epoch "
            f"{result['details']['best_epoch']}, {time.time() - started:.0f} s")

    if args.no_join:
        return
    results = []
    for seed in args.seeds:
        for fold in range(config.N_FOLDS):
            part = part_path(fold, seed)
            if not os.path.exists(part):
                log(f"{name}: fold {fold} seed {seed} is not done yet, the joined file is not written")
                return
            with np.load(part) as saved:
                results.append({"seed": seed, "fold": fold, "test": saved["test"], "test_index": saved["test_index"],
                                "val_accuracy": float(saved["val_accuracy"]),
                                "train_accuracy": float(saved["train_accuracy"]), "n_train": int(saved["n_train"]),
                                "details": {"best_epoch": int(saved["best_epoch"]),
                                            "val_accuracy_per_epoch": saved["history"].tolist()}})
    seeds = sorted(set(r["seed"] for r in results))
    proba, val, train, runs = experiments.assemble(results, len(cache.clips), seeds)
    info = {"experiment": name, "model": "finetune", "input": args.encoder, "arm": args.arm, "seeds": seeds,
            "encoder": pretrained.ENCODERS[args.encoder]["model"], "epochs": args.epochs, "lr": args.lr,
            "batch_size": args.batch_size, "runs": runs}
    evaluation.save_predictions(os.path.join(args.results_dir, "predictions"), name, proba, val, train, info)
    accuracy = (proba.argmax(axis=2) == cache.labels[None, :]).mean(axis=1)
    log(f"{name}: test clip accuracy per seed {np.round(accuracy, 4).tolist()}, validation {val.mean():.4f}")


if __name__ == "__main__":
    main()
