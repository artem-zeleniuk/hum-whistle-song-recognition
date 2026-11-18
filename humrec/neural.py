"""Sequence models in PyTorch and one training loop for all of them."""

import copy

import numpy as np
import torch
from torch import nn

from . import config

# Keras defaults of batch normalisation (running-average weight 0.99, epsilon 0.001), kept so that the
# networks behave like the Keras models they were first written as.
_BN = {"momentum": 0.01, "eps": 1e-3}


class Conv1dNet(nn.Module):
    """Four 1D convolutions along time. Input (batch, frames, channels)."""

    def __init__(self, n_channels, n_classes=config.N_CLASSES):
        super().__init__()

        def block(c_in, c_out, kernel, dropout, pool=True):
            layers = [nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2), nn.ReLU(), nn.BatchNorm1d(c_out, **_BN)]
            if pool:
                layers += [nn.MaxPool1d(2), nn.Dropout(dropout)]
            return layers

        self.features = nn.Sequential(*block(n_channels, 32, 7, 0.2), *block(32, 64, 5, 0.2),
                                      *block(64, 128, 3, 0.3), *block(128, 128, 3, 0.0, pool=False))
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, x):
        x = self.features(x.transpose(1, 2))
        return self.head(x.mean(dim=2))


class Conv2dNet(nn.Module):
    """The same layout with 2D convolutions over time and feature index. Input (batch, frames, channels)."""

    def __init__(self, n_channels, n_classes=config.N_CLASSES):
        super().__init__()
        del n_channels  # the feature axis is a spatial axis here, so its length does not set a layer size

        def block(c_in, c_out, kernel, dropout, pool=True):
            padding = (kernel[0] // 2, kernel[1] // 2)
            layers = [nn.Conv2d(c_in, c_out, kernel, padding=padding), nn.ReLU(), nn.BatchNorm2d(c_out, **_BN)]
            if pool:
                layers += [nn.MaxPool2d(2), nn.Dropout(dropout)]
            return layers

        self.features = nn.Sequential(*block(1, 32, (7, 3), 0.2), *block(32, 64, (5, 3), 0.2),
                                      *block(64, 128, (3, 3), 0.3), *block(128, 128, (3, 3), 0.0, pool=False))
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, x):
        x = self.features(x.unsqueeze(1))
        return self.head(x.mean(dim=(2, 3)))


class KeyInvariantNet(nn.Module):
    """A network whose output does not change when the melody is moved to another key.

    Input (batch, frames, channels) in the `melody` layout (12 chroma values, the onset strength and the
    tempo, 14 channels) or the `melody_full` layout (the same with the 12 frame-to-frame differences of
    the chroma after the chroma, 26 channels). Moving a melody by k semitones rotates the 12 chroma
    values of every frame by k places. The chroma (and its difference) is therefore arranged as an image
    over time and the circle of pitch classes, the onset strength is added as one more image that is the
    same for all pitch classes, and the convolutions pad the pitch-class axis circularly. A rotation of
    the input then only rotates every feature map, and the mean over the pitch classes at the end removes
    it. Pooling between the blocks is along time only. The tempo joins after the pooling.

    `pitch_kernel` is the width of the convolutions along the circle: 3 sees only neighbouring semitones
    in one layer, 12 sees every interval at once (a full convolution over the group of the 12 keys).
    """

    PITCH_CLASSES = 12

    def __init__(self, n_channels=26, n_classes=config.N_CLASSES, pitch_kernel=3):
        super().__init__()
        if n_channels not in (14, 26):
            raise ValueError("the key-invariant network expects the melody (14) or melody_full (26) input")
        self.pitch_images = (n_channels - 2) // self.PITCH_CLASSES     # 1 or 2 images of 12 pitch classes

        def block(c_in, c_out, time_kernel, dropout, pool=True):
            layers = [_CircularConv(c_in, c_out, time_kernel, pitch_kernel), nn.ReLU(), nn.BatchNorm2d(c_out, **_BN)]
            if pool:
                layers += [nn.MaxPool2d((2, 1)), nn.Dropout(dropout)]
            return layers

        self.features = nn.Sequential(*block(self.pitch_images + 1, 32, 7, 0.2), *block(32, 64, 5, 0.2),
                                      *block(64, 128, 3, 0.3), *block(128, 128, 3, 0.0, pool=False))
        self.head = nn.Sequential(nn.Linear(128 + 1, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, x):
        n, k = self.PITCH_CLASSES, self.pitch_images
        images = [x[:, :, i * n:(i + 1) * n] for i in range(k)]                  # chroma, and its difference
        images.append(x[:, :, k * n:k * n + 1].expand(-1, -1, n))                # onset strength
        tempo = x[:, 0, k * n + 1:]
        pooled = self.features(torch.stack(images, dim=1)).mean(dim=(2, 3))      # (batch, images, frames, 12) in
        return self.head(torch.cat([pooled, tempo], dim=1))


class _CircularConv(nn.Module):
    """A convolution `pitch_kernel` wide along the pitch classes (circular padding) and `time_kernel` long in time."""

    def __init__(self, c_in, c_out, time_kernel, pitch_kernel=3):
        super().__init__()
        self.pad = ((pitch_kernel - 1) // 2, pitch_kernel // 2, 0, 0)     # the width of 12 pitch classes is kept
        self.conv = nn.Conv2d(c_in, c_out, (time_kernel, pitch_kernel), padding=(time_kernel // 2, 0))

    def forward(self, x):
        return self.conv(nn.functional.pad(x, self.pad, mode="circular"))


class KeyInvariantNetFull(KeyInvariantNet):
    """The key-invariant network with convolutions over the whole circle of pitch classes."""

    def __init__(self, n_channels=26, n_classes=config.N_CLASSES):
        super().__init__(n_channels, n_classes, pitch_kernel=12)


def pitch_class_channels(n_channels):
    """Channel ranges that hold one value per pitch class in the `melody` (14) and `melody_full` (26) layouts."""
    return tuple((i * 12, (i + 1) * 12) for i in range((n_channels - 2) // 12))


def transpose_key(batch, rng, groups=None):
    """Move every sequence of a batch to a random key.

    `groups` are the channel ranges that hold one value per pitch class (by default those of the layout,
    see `pitch_class_channels`). Each sequence gets its own rotation by 0 to 11 places, the other
    channels stay as they are.
    """
    groups = pitch_class_channels(batch.shape[2]) if groups is None else groups
    shift = torch.as_tensor(rng.integers(0, 12, size=batch.shape[0]), device=batch.device)
    index = (torch.arange(12, device=batch.device)[None, :] - shift[:, None]) % 12
    index = index[:, None, :].expand(-1, batch.shape[1], -1)
    out = batch.clone()
    for start, stop in groups:
        out[:, :, start:stop] = torch.gather(batch[:, :, start:stop], 2, index)
    return out


class BiLSTMNet(nn.Module):
    """One bidirectional LSTM layer with 32 units per direction and a linear layer on its two final states."""

    def __init__(self, n_channels, n_classes=config.N_CLASSES, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden, n_classes)

    def forward(self, x):
        _, (state, _) = self.lstm(x)
        return self.fc(torch.cat([state[0], state[1]], dim=1))


class BiLSTMPoolNet(nn.Module):
    """The same LSTM with the mean of its outputs over time as the summary of the clip."""

    def __init__(self, n_channels, n_classes=config.N_CLASSES, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden, n_classes)

    def forward(self, x):
        outputs, _ = self.lstm(x)
        return self.fc(outputs.mean(dim=1))


class MLPHead(nn.Module):
    """Classifier on a fixed embedding: 768 to 256 to the classes. Input (batch, 768)."""

    def __init__(self, n_inputs=768, n_classes=config.N_CLASSES):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.3),
                                        nn.Linear(256, n_classes))

    def forward(self, x):
        return self.classifier(x)


NETWORKS = {"cnn1d": Conv1dNet, "cnn2d": Conv2dNet, "keynet": KeyInvariantNet, "keynet12": KeyInvariantNetFull,
            "bilstm": BiLSTMNet, "bilstm_pool": BiLSTMPoolNet, "mlp": MLPHead}

# Training settings per kind of network.
SEQUENCE_SETTINGS = {"lr": 5e-4, "batch_size": 32, "max_epochs": 100, "patience": 15, "lr_patience": 7}
HEAD_SETTINGS = {"lr": 1e-3, "batch_size": 8, "max_epochs": 30, "patience": 7, "lr_patience": 3}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def standardise(x_train, *others, pooled=()):
    """Scale every channel (last axis) to zero mean and unit variance of the training set.

    The channels of each range in `pooled` share one mean and one standard deviation. This is needed
    when the channels of a range are exchanged with each other (the 12 pitch classes under a change of
    key), because separate scales would make the result depend on the key.
    """
    flat = x_train.reshape(-1, x_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    for start, stop in pooled:
        mean[start:stop] = flat[:, start:stop].mean()
        std[start:stop] = flat[:, start:stop].std()
    std = np.where(std > 0, std, 1.0)
    return tuple(((x - mean) / std).astype(np.float32) for x in (x_train,) + others)


def stretch_in_time(batch, factor, rng):
    """Play a batch of sequences (batch, frames, channels) `factor` times slower and keep the length.

    The sequences are interpolated to `factor` times their length. A longer result is cut at a random
    position, a shorter one is padded with zeros at the end. For a melody this changes the tempo and
    leaves the notes as they are.
    """
    frames = batch.shape[1]
    new_frames = max(8, int(round(frames * factor)))
    out = torch.nn.functional.interpolate(batch.transpose(1, 2), size=new_frames, mode="linear", align_corners=False)
    out = out.transpose(1, 2)
    if new_frames >= frames:
        start = int(rng.integers(0, new_frames - frames + 1))
        return out[:, start:start + frames, :]
    return torch.nn.functional.pad(out, (0, 0, 0, frames - new_frames))


@torch.no_grad()
def predict_proba(model, x, batch_size=256):
    """Class probabilities for a tensor that already lives on the device of the model."""
    model.eval()
    parts = [torch.softmax(model(x[i:i + batch_size]), dim=1) for i in range(0, len(x), batch_size)]
    return torch.cat(parts).float().cpu().numpy()


def fit_network(kind, x_train, y_train, x_val, y_val, x_test, seed, device="cpu", lr=5e-4, batch_size=32,
                max_epochs=100, patience=15, lr_patience=7, lr_factor=0.5, min_lr=1e-6, label_smoothing=0.0,
                mixup=0.0, stretch=0.0, transpose=False):
    """Train one network and return its class probabilities.

    Adam with cross-entropy. After every epoch the validation loss and accuracy are measured. The
    learning rate is multiplied by `lr_factor` when the validation loss has not improved for `lr_patience`
    epochs. Training stops when the validation accuracy has not improved for `patience` epochs, and the
    weights of the best epoch are restored.

    Two optional regularisers for noisy data: `label_smoothing` moves that share of the target from the
    true class to all classes, and `mixup` (the parameter of a Beta distribution, 0 switches it off) trains
    on convex combinations of two inputs of a batch with the same combination of their targets.
    `stretch` (sequence inputs only, 0 switches it off) plays every training batch faster or slower by a
    random factor between 1 / (1 + stretch) and 1 + stretch, see `stretch_in_time`. `transpose` (inputs in
    the `melody_full` layout only) moves every training sequence to a random key, see `transpose_key`.

    Returns a dict with `val` and `test` (class probabilities), `train_accuracy`, `epochs` (epochs run),
    `best_epoch` and `history` (validation accuracy per epoch).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    model = NETWORKS[kind](x_train.shape[-1]).to(device)
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=device)
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(y_val, dtype=torch.long, device=device)
    x_test_t = torch.as_tensor(x_test, dtype=torch.float32, device=device)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=lr_factor,
                                                           patience=lr_patience, min_lr=min_lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    order = torch.Generator().manual_seed(seed)
    mixing = np.random.default_rng(seed)
    best_accuracy, best_epoch, best_state, history = -1.0, 0, None, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(x_train_t), generator=order).to(device)
        for start in range(0, len(permutation), batch_size):
            index = permutation[start:start + batch_size]
            if len(index) < 2:
                continue  # batch normalisation needs more than one clip
            optimiser.zero_grad()
            if mixup > 0:
                share = float(mixing.beta(mixup, mixup))
                partner = index[torch.randperm(len(index), generator=order).to(device)]
                logits = model(share * x_train_t[index] + (1.0 - share) * x_train_t[partner])
                loss = (share * train_loss(logits, y_train_t[index])
                        + (1.0 - share) * train_loss(logits, y_train_t[partner]))
            elif stretch > 0 or transpose:
                batch = x_train_t[index]
                if transpose:
                    batch = transpose_key(batch, mixing)
                if stretch > 0:
                    factor = float(np.exp(mixing.uniform(-np.log1p(stretch), np.log1p(stretch))))
                    batch = stretch_in_time(batch, factor, mixing)
                loss = train_loss(model(batch), y_train_t[index])
            else:
                loss = train_loss(model(x_train_t[index]), y_train_t[index])
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(x_val_t[i:i + 256]) for i in range(0, len(x_val_t), 256)])
            val_loss = float(loss_fn(logits, y_val_t))
            val_accuracy = float((logits.argmax(dim=1) == y_val_t).float().mean())
        history.append(val_accuracy)
        scheduler.step(val_loss)
        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    train = predict_proba(model, x_train_t)
    return {
        "val": predict_proba(model, x_val_t),
        "test": predict_proba(model, x_test_t),
        "train_accuracy": float(np.mean(train.argmax(axis=1) == np.asarray(y_train))),
        "epochs": len(history),
        "best_epoch": best_epoch,
        "history": history,
    }
