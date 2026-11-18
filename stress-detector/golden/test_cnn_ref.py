"""Checks the integer golden model against a plain float computation."""
import numpy as np

from cnn_ref import N_FEAT, P, conv_relu_requant, forward, maxpool, pixel_to_int8
from config import C_OUT, IMG, K, N_CLASSES


def test_pixel_mapping_matches_msb_flip():
    p = np.arange(256)
    flipped = ((p ^ 0x80).astype(np.uint8)).astype(np.int8)
    assert np.array_equal(pixel_to_int8(p), flipped)


def test_conv_matches_direct_loops():
    rng = np.random.default_rng(0)
    x = pixel_to_int8(rng.integers(0, 256, (IMG, IMG)))
    w = rng.integers(-128, 128, (C_OUT, K, K))
    b = rng.integers(-1000, 1000, C_OUT)
    y = conv_relu_requant(x, w, b, 8)
    for c, i, j in [(0, 0, 0), (1, 10, 20), (3, IMG - K, IMG - K)]:
        acc = int((x[i:i + K, j:j + K] * w[c]).sum() + b[c])
        assert y[c, i, j] == min(max((acc + 128) >> 8, 0), 127)


def test_maxpool_shape_and_values():
    fm = np.arange(C_OUT * 46 * 46).reshape(C_OUT, 46, 46)
    pooled = maxpool(fm)
    assert pooled.shape == (C_OUT, P, P)
    assert pooled[0, 0, 0] == fm[0, 1, 1]


def test_forward_runs_and_logits_are_exact():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (IMG, IMG)).astype(np.uint8)
    w_conv = rng.integers(-128, 128, (C_OUT, K, K))
    b_conv = rng.integers(-1000, 1000, C_OUT)
    w_fc = rng.integers(-128, 128, (N_CLASSES, N_FEAT))
    b_fc = rng.integers(-1000, 1000, N_CLASSES)
    cls, logits, feat = forward(img, w_conv, b_conv, 8, w_fc, b_fc)
    manual = [int(sum(int(f) * int(w) for f, w in zip(feat.ravel(), w_fc[j]))) + int(b_fc[j])
              for j in range(N_CLASSES)]
    assert list(logits) == manual
    assert cls == int(np.argmax(manual))
