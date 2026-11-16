"""Reference (slow, obviously correct) expectimax for validating the torch teacher.

Not used in training: it exists so test/test_sync.py can check the vectorised torch
search picks the same moves as this straightforward recursive implementation.
"""
from __future__ import annotations

import numpy as np

from src import engine


def weight_matrix(w) -> np.ndarray:
    if w.matrix == "snake":
        return np.array([[16.0, 8.0, 4.0, 2.0], [2.0, 4.0, 8.0, 16.0], [4.0, 2.0, 1.0, 0.5],
                         [2.0, 1.0, 0.5, 0.25]])
    if w.matrix == "nneonneo":
        return np.array([[65536.0, 4096.0, 256.0, 16.0], [32768.0, 2048.0, 128.0, 8.0],
                         [1024.0, 64.0, 4.0, 4.0], [32.0, 2.0, 2.0, 2.0]])
    rows = [w.base * (w.row_ratio ** (3 - i)) / (w.row_ratio ** 3) for i in range(4)]
    cols = [(w.col_ratio ** (3 - j)) / (w.col_ratio ** 3) for j in range(4)]
    return np.array(rows)[:, None] * np.array(cols)[None, :]


def rot(a: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(a, k, axes=(0, 1))


def heuristic(board: np.ndarray, w) -> float:
    """Mirrors teacher.heuristic for a single board."""
    B = np.asarray(board)
    logv = np.where(B > 0, np.log2(np.maximum(B, 1)), 0.0)
    if w.tile_score == "value":
        tile = B.astype(np.float64)
    elif w.tile_score == "log2":
        tile = logv * logv
    else:
        tile = logv
    score = w.empty * float((B == 0).sum())
    W = weight_matrix(w)
    best = 0.0
    for k in range(w.rotations):
        best = max(best, float((rot(tile, k) * rot(W, k)).sum()))
    score += w.corner * best + w.max_tile * float(np.log2(max(B.max(), 1)))
    dh = float(np.abs(logv[:, :-1] - logv[:, 1:]).sum())
    dv = float(np.abs(logv[:-1, :] - logv[1:, :]).sum())
    score -= w.smooth * (dh + dv)
    vh = float(((logv[:, :-1] > logv[:, 1:]) & (B[:, :-1] != 0) & (B[:, 1:] != 0)).sum())
    vv = float(((logv[:-1, :] > logv[1:, :]) & (B[:-1, :] != 0) & (B[1:, :] != 0)).sum())
    score -= w.mono * (vh + vv)
    return score


def expectimax(board: np.ndarray, depth: int, w) -> tuple[float, int]:
    """Exact chance nodes: every empty cell x {2, 4}."""
    if engine.dead_boards(board[None])[0]:
        return heuristic(board, w), -1  # game over: worth its potential, nothing more
    if depth == 0:
        return heuristic(board, w), -1
    best_val, best_move = -np.inf, 0
    for d in range(4):
        child = engine.move_boards(board[None], d)[0]
        if np.array_equal(child, board):
            continue
        vals = []
        for (i, j) in np.argwhere(child == 0):
            for v in (2, 4):
                b2 = child.copy()
                b2[i, j] = v
                vals.append(expectimax(b2, depth - 1, w)[0])
        val = float(np.mean(vals))
        if val > best_val:
            best_val, best_move = val, d
    return best_val, best_move
