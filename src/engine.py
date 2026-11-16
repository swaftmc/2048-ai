"""2048 board rules in numpy.

Board layout: int array shape (4, 4) or batched (N, 4, 4). Tiles are powers of two;
0 means empty. Directions are 0=left, 1=right, 2=up, 3=down.
"""
from __future__ import annotations

import numpy as np

SIZE = 4
CELLS = SIZE * SIZE
LEFT, RIGHT, UP, DOWN = 0, 1, 2, 3
DIR_NAMES = ("left", "right", "up", "down")
SPAWN_P4 = 0.10  # 10% of spawns are a 4, the rest are 2


def _compact_left(row: np.ndarray) -> np.ndarray:
    """Push zeros to the right, preserving tile order."""
    nonzero = row != 0
    order = np.argsort(~nonzero, axis=-1, kind="stable")
    return np.take_along_axis(row, order, axis=-1)


def _merge_left(row: np.ndarray) -> np.ndarray:
    """Slide tiles left and merge equal neighbours, scanning left to right.

    A merged tile cannot merge again in the same move, and the scan restarts after
    the pair it just consumed - so [2,8,8,4] becomes [2,16,4,0], not [2,8,8,4].
    """
    x = _compact_left(row)
    x0, x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    m01 = (x0 == x1) & (x0 != 0)
    m12 = (x1 == x2) & (x1 != 0) & ~m01
    m23 = (x2 == x3) & (x2 != 0) & ~m12
    o0 = np.where(m01, 2 * x0, x0)
    o1 = np.where(m01, np.where(m23, 2 * x2, x2), np.where(m12, 2 * x1, x1))
    o2 = np.where(m01, np.where(m23, 0, x3), np.where(m12, x3, np.where(m23, 2 * x2, x2)))
    o3 = np.where(m01 | m12 | m23, 0, x3)
    return np.stack([o0, o1, o2, o3], axis=-1)


def move_boards(B: np.ndarray, d: int) -> np.ndarray:
    """Apply move d to a batch of boards."""
    B = np.asarray(B)
    if d == LEFT:
        return _merge_left(B)
    if d == RIGHT:
        return _merge_left(np.ascontiguousarray(B[:, :, ::-1]))[:, :, ::-1]
    if d == UP:
        return _merge_left(np.swapaxes(B, 1, 2)).swapaxes(1, 2)
    if d == DOWN:
        # transpose -> reverse each row (= reverse each column of B) -> merge left
        # (= pack each column downwards) -> transpose back -> reverse rows again
        Bt = np.ascontiguousarray(np.swapaxes(B, 1, 2)[:, :, ::-1])
        return np.ascontiguousarray(np.swapaxes(_merge_left(Bt), 1, 2)[:, ::-1, :])
    raise ValueError(f"bad direction {d}")


def changed_moves(B: np.ndarray) -> np.ndarray:
    """Boolean (N, 4): did move d change the board?"""
    B = np.asarray(B)
    return np.stack([(move_boards(B, d) != B).any(axis=(1, 2)) for d in range(4)], axis=1)


def empty_mask(B: np.ndarray) -> np.ndarray:
    B = np.asarray(B)
    return B == 0


def spawn_boards(B: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Place one random tile (90% a 2, 10% a 4) in a random empty cell of each board."""
    B = np.asarray(B)
    N = B.shape[0]
    flat = B.reshape(N, CELLS)
    empty = flat == 0
    count = empty.sum(axis=1)
    safe = np.maximum(count, 1)
    k = np.minimum((rng.random(N) * safe).astype(np.int64), safe - 1)
    cum = np.cumsum(empty, axis=1)
    pos = np.argmax(cum == (k + 1)[:, None], axis=1)
    val = np.where(rng.random(N) < SPAWN_P4, 4, 2)
    out = flat.copy()
    has = count > 0
    rows = np.nonzero(has)[0]
    out[rows, pos[rows]] = val[rows]
    return out.reshape(B.shape)


def dead_boards(B: np.ndarray) -> np.ndarray:
    """Boolean (N,): no empty cell and no mergeable pair -> game over."""
    B = np.asarray(B)
    has_empty = (B == 0).any(axis=(1, 2))
    eq_h = (B[:, :, :-1] == B[:, :, 1:]) & (B[:, :, :-1] != 0)
    eq_v = (B[:, :-1, :] == B[:, 1:, :]) & (B[:, :-1, :] != 0)
    can_merge = eq_h.any(axis=(1, 2)) | eq_v.any(axis=(1, 2))
    return ~has_empty & ~can_merge


def has_tile(B: np.ndarray, value: int) -> np.ndarray:
    B = np.asarray(B)
    return (B == value).any(axis=(1, 2))


def max_tile(B: np.ndarray) -> np.ndarray:
    B = np.asarray(B)
    return B.max(axis=(1, 2))


def new_board(rng: np.random.Generator) -> np.ndarray:
    """Fresh game: two random tiles on an empty board."""
    B = np.zeros((1, SIZE, SIZE), dtype=np.int64)
    B = spawn_boards(B, rng)
    B = spawn_boards(B, rng)
    return B[0]


def clone(B: np.ndarray) -> np.ndarray:
    return np.array(B, dtype=np.int64, copy=True)
