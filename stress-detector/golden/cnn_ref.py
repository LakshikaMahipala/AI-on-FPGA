"""Bit-exact Python reference for the stress-detector CNN core (rtl/cnn_core.sv).

Network (sizes from config.py):
    48x48 grayscale image (uint8)
      -> x = pixel - 128                      (int8)
      -> conv 3x3, C_OUT filters, valid       (int32 accumulate, + bias)
      -> requant: clip(round(acc / 2^SHIFT), 0, 127)   (ReLU folded into the clip)
      -> max-pool 2x2                         (C_OUT x 23 x 23 int8 features)
      -> fully connected -> N_CLASSES logits  (int32, + bias)
      -> argmax                               (0 = not stressed, 1 = stressed)

The hardware computes exactly these integers, so the testbenches demand an
exact match.
"""
from __future__ import annotations

import numpy as np

from config import C_OUT, IMG, K, N_CLASSES, POOL

P = (IMG - K + 1) // POOL          # pooled feature map size (23)
N_FEAT = C_OUT * P * P             # FC input length (2116)


def rshift_round(x, shift: int):
    x = np.asarray(x, dtype=np.int64)
    return x if shift <= 0 else (x + (1 << (shift - 1))) >> shift


def pixel_to_int8(img_u8: np.ndarray) -> np.ndarray:
    """uint8 pixel -> int8 value pixel - 128 (in hardware: invert the MSB)."""
    return img_u8.astype(np.int64) - 128


def conv_relu_requant(x: np.ndarray, w: np.ndarray, b: np.ndarray, shift: int) -> np.ndarray:
    """x: (IMG, IMG) int, w: (C_OUT, K, K), b: (C_OUT,) -> (C_OUT, IMG-K+1, IMG-K+1) in [0, 127]."""
    out_size = IMG - K + 1
    acc = np.zeros((C_OUT, out_size, out_size), dtype=np.int64) + b.astype(np.int64)[:, None, None]
    for ky in range(K):
        for kx in range(K):
            acc += w[:, ky, kx].astype(np.int64)[:, None, None] * \
                   x[None, ky:ky + out_size, kx:kx + out_size]
    return np.clip(rshift_round(acc, shift), 0, 127)


def maxpool(fm: np.ndarray) -> np.ndarray:
    c, h, w = fm.shape
    return fm[:, :P * POOL, :P * POOL].reshape(c, P, POOL, P, POOL).max(axis=(2, 4))


def forward(img_u8, w_conv, b_conv, shift, w_fc, b_fc):
    """Returns (predicted class, logits, pooled features)."""
    feat = maxpool(conv_relu_requant(pixel_to_int8(img_u8), w_conv, b_conv, shift))
    logits = w_fc.astype(np.int64).reshape(N_CLASSES, N_FEAT) @ feat.reshape(-1) + b_fc.astype(np.int64)
    assert np.abs(logits).max() < 2**31, "FC accumulator would overflow int32"
    return int(np.argmax(logits)), logits, feat   # argmax: lowest index wins ties, as in RTL


def expected_cycles() -> int:
    """Cycle count of cnn_core from the cycle after `start` is sampled to `done`."""
    conv = C_OUT * P * P * (POOL * POOL) * (K * K + 1)   # K*K MACs + 1 activation step per window
    fc = N_CLASSES * (N_FEAT + 1)                         # N_FEAT MACs + 1 store step per class
    return conv + fc + 1                                  # + 1 argmax / finish step
