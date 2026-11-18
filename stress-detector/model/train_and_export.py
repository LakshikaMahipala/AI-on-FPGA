"""Train the FPGA-sized stress CNN on FER2013 and export INT8 weights for the hardware.

The VGG-16 in notebooks/ (138 M parameters) is a software baseline; it cannot
fit in a small FPGA. This network has 4,274 parameters and matches
rtl/cnn_core.sv exactly:

    48x48 gray -> conv3x3 (4 filters) -> ReLU -> maxpool 2x2 -> FC (2116 -> 2)

Labels (same as the notebook): angry, disgust, fear, sad -> stressed (1);
happy, neutral -> not stressed (0); surprise is skipped.

Usage (GPU optional, e.g. Colab):
    python model/train_and_export.py --data fer2013/train --epochs 15
Writes model/exported_weights.npz and prints float vs integer accuracy.
Then:  WEIGHTS=model/exported_weights.npz make sim
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "golden"))
from cnn_ref import forward  # noqa: E402  (bit-exact integer model)

STRESSED = {"angry", "disgust", "fear", "sad"}
NOT_STRESSED = {"happy", "neutral"}


class StressFER(Dataset):
    """FER2013 folders -> (uint8 48x48 image tensor, stress label)."""

    def __init__(self, root: str):
        tf = transforms.Compose([transforms.Grayscale(), transforms.Resize((48, 48)),
                                 transforms.PILToTensor()])
        base = datasets.ImageFolder(root, transform=tf)
        self.base = base
        self.items = [(i, 1 if base.classes[c].lower() in STRESSED else 0)
                      for i, (_, c) in enumerate(base.samples)
                      if base.classes[c].lower() in STRESSED | NOT_STRESSED]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, y = self.items[k]
        img_u8, _ = self.base[i]
        return img_u8, y


def to_model_input(img_u8: torch.Tensor) -> torch.Tensor:
    """Same mapping as the hardware: (pixel - 128) / 128."""
    return (img_u8.float() - 128.0) / 128.0


class TinyStressCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(4 * 23 * 23, 2)

    def features(self, x):
        return torch.relu(self.conv(x))

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.features(x)), 1))


def quantize_symmetric(t: np.ndarray):
    scale = float(np.abs(t).max()) / 127 or 1.0
    return np.clip(np.round(t / scale), -127, 127).astype(np.int64), scale


@torch.no_grad()
def export(model: TinyStressCNN, calib_loader) -> dict:
    x_scale = 1 / 128
    w_conv, s_wc = quantize_symmetric(model.conv.weight[:, 0].cpu().numpy())
    acc_scale = x_scale * s_wc
    b_conv = np.round(model.conv.bias.cpu().numpy() / acc_scale).astype(np.int64)

    act_max = max(float(model.features(to_model_input(x)).max()) for x, _ in calib_loader)
    shift = max(1, math.ceil(math.log2(act_max / (127 * acc_scale))))
    y_scale = acc_scale * 2 ** shift

    w_fc, s_wf = quantize_symmetric(model.fc.weight.cpu().numpy())
    b_fc = np.round(model.fc.bias.cpu().numpy() / (y_scale * s_wf)).astype(np.int64)
    return dict(w_conv=w_conv, b_conv=b_conv, shift=shift, w_fc=w_fc, b_fc=b_fc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="FER2013 train folder with one sub-folder per emotion")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default="model/exported_weights.npz")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds = StressFER(args.data)
    n_val = len(ds) // 5
    train, val = random_split(ds, [len(ds) - n_val, n_val])
    train_dl = DataLoader(train, batch_size=128, shuffle=True, num_workers=2)
    val_dl = DataLoader(val, batch_size=256, num_workers=2)

    model = TinyStressCNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_dl:
            loss = nn.functional.cross_entropy(model(to_model_input(x).to(dev)), y.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"epoch {epoch + 1}: loss {loss.item():.3f}")

    model = model.cpu().eval()
    q = export(model, [b for _, b in zip(range(20), train_dl)])
    np.savez(args.out, **q)

    float_ok = int_ok = n = 0
    with torch.no_grad():
        for x, y in val_dl:
            float_ok += int((model(to_model_input(x)).argmax(1) == y).sum())
            for img, label in zip(x[:, 0].numpy().astype(np.uint8), y.numpy()):
                cls, _, _ = forward(img, q["w_conv"], q["b_conv"], q["shift"], q["w_fc"], q["b_fc"])
                int_ok += int(cls == label)
            n += len(y)
    print(f"validation accuracy: float {float_ok / n:.3f} | INT8 hardware model {int_ok / n:.3f} "
          f"({n} images, shift={q['shift']})")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
