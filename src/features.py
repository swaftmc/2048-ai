"""Board features for the linear student.

`features(board)` returns a float64 vector of length 72 for a single (4,4) board or
an (N,72) matrix for a batch. The JS mirror lives in web/index.html; test/test_sync.py
asserts the two implementations agree to 1e-9.
"""
from __future__ import annotations

import numpy as np

LOG_MAX = 11.0  # log2(2048), the tile we are aiming for
N_FEATURES = 104
# Corner-anchored weight matrix; the heavy corner is (0,0).
W_CORNER = np.array(
    [
        [16.0, 8.0, 4.0, 2.0],
        [8.0, 4.0, 2.0, 1.0],
        [4.0, 2.0, 1.0, 0.5],
        [2.0, 1.0, 0.5, 0.25],
    ]
)
# The position weights the teacher itself uses. Multiplying raw tile values by these
# makes the teacher's evaluation a linear function of the features, so the ridge head
# can imitate the teacher's judgement instead of guessing it.
W_NNEONNE = np.array(
    [
        [65536.0, 4096.0, 256.0, 16.0],
        [32768.0, 2048.0, 128.0, 8.0],
        [1024.0, 64.0, 4.0, 4.0],
        [32.0, 2.0, 2.0, 2.0],
    ]
)
BIG_TILE = 16  # tiles below this are ignored by the teacher's evaluation
EVAL_SCALE = 1.0e6

_EDGE_MASK = np.zeros((4, 4), dtype=bool)
_EDGE_MASK[0, :] = _EDGE_MASK[3, :] = True
_EDGE_MASK[:, 0] = _EDGE_MASK[:, 3] = True
_EDGE_MASK[0, 0] = _EDGE_MASK[0, 3] = _EDGE_MASK[3, 0] = _EDGE_MASK[3, 3] = False


def _log2(B: np.ndarray) -> np.ndarray:
    return np.where(B > 0, np.log2(np.maximum(B, 1)), 0.0)


def features(B: np.ndarray) -> np.ndarray:
    """(4,4) -> (72,), or (N,4,4) -> (N,72)."""
    B = np.asarray(B)
    single = B.ndim == 2
    if single:
        B = B[None, ...]
    n = B.shape[0]
    logv = _log2(B)
    mx = B.max(axis=(1, 2))
    mx3 = mx[:, None, None]

    cell_log = (logv / LOG_MAX).reshape(n, 16)
    cell_empty = (B == 0).astype(np.float64).reshape(n, 16)
    cell_ismax = ((B == mx3) & (B > 0)).astype(np.float64).reshape(n, 16)

    row_log = logv.sum(axis=2) / (4.0 * LOG_MAX)
    col_log = logv.sum(axis=1) / (4.0 * LOG_MAX)

    log2max = np.log2(np.maximum(mx, 1)) / LOG_MAX
    empty_frac = (B == 0).sum(axis=(1, 2)) / 16.0

    diff_h = np.abs(logv[:, :, :-1] - logv[:, :, 1:])
    pair_h = (B[:, :, :-1] != 0) & (B[:, :, 1:] != 0)
    diff_v = np.abs(logv[:, :-1, :] - logv[:, 1:, :])
    pair_v = (B[:, :-1, :] != 0) & (B[:, 1:, :] != 0)
    n_pairs = pair_h.sum(axis=(1, 2)) + pair_v.sum(axis=(1, 2))
    smoothness = (diff_h * pair_h).sum(axis=(1, 2)) + (diff_v * pair_v).sum(axis=(1, 2))
    smoothness = smoothness / np.maximum(n_pairs, 1) / LOG_MAX

    viol_h = (logv[:, :, :-1] > logv[:, :, 1:]) & pair_h
    viol_v = (logv[:, :-1, :] > logv[:, 1:, :]) & pair_v
    mono = (viol_h.sum(axis=(1, 2)) + viol_v.sum(axis=(1, 2))) / 24.0

    eq_h = (B[:, :, :-1] == B[:, :, 1:]) & (B[:, :, :-1] != 0)
    eq_v = (B[:, :-1, :] == B[:, 1:, :]) & (B[:, :-1, :] != 0)
    mergeable = (eq_h.sum(axis=(1, 2)) + eq_v.sum(axis=(1, 2))) / 24.0

    counts = np.stack(
        [(B >= t).sum(axis=(1, 2)) / 16.0 for t in (32, 64, 128, 256)], axis=1
    )

    corners = np.stack([B[:, 0, 0], B[:, 0, 3], B[:, 3, 0], B[:, 3, 3]], axis=1)
    max_in_corner = ((corners == mx[:, None]) & (mx[:, None] > 0)).any(axis=1).astype(np.float64)
    max_on_edge = ((B == mx3) & _EDGE_MASK & (mx3 > 0)).any(axis=(1, 2)).astype(np.float64)

    corner_w = np.empty((n, 4), dtype=np.float64)
    grads = np.empty((n, 4), dtype=np.float64)
    for k in range(4):
        logvk = np.rot90(logv, k, axes=(1, 2))
        Wk = np.rot90(W_CORNER, k, axes=(0, 1))
        corner_w[:, k] = (logvk * Wk).sum(axis=(1, 2)) / (LOG_MAX * 16.0)
        rows = logvk.sum(axis=2)
        grads[:, k] = np.maximum(rows[:, :-1] - rows[:, 1:], 0.0).sum(axis=1) / (4.0 * LOG_MAX)
    best_grad = grads.max(axis=1)

    # teacher-evaluation mirrors: raw tile value x the teacher's position weights
    vals = B.astype(np.float64)
    nneon_all = (vals * W_NNEONNE).reshape(n, 16) / EVAL_SCALE
    nneon_big = (np.where(B >= BIG_TILE, vals, 0.0) * W_NNEONNE).reshape(n, 16) / EVAL_SCALE

    out = np.concatenate(
        [
            cell_log,
            cell_empty,
            cell_ismax,
            row_log,
            col_log,
            log2max[:, None],
            empty_frac[:, None],
            smoothness[:, None],
            mono[:, None],
            mergeable[:, None],
            counts,
            max_in_corner[:, None],
            max_on_edge[:, None],
            corner_w,
            best_grad[:, None],
            nneon_all,
            nneon_big,
        ],
        axis=1,
    )
    assert out.shape == (n, N_FEATURES), out.shape
    return out[0] if single else out
