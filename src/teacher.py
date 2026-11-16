"""Depth-limited expectimax teacher (torch).

The teacher is a search, not a learner: it plays 2048 with a hand-written heuristic
evaluation and a chance-sampled expectimax tree. It exists to label the student
(what move is best, how many moves until a 2048 tile appears).

Batched: all games in a batch share the same vectorised tree expansion, so the whole
thing runs on a GPU (Colab T4) or CPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src import engine

INVALID = -1.0e7  # pseudo-value for a move that does nothing
TARGET_TILE = 2048
FAIL_LABEL = 500.0  # moves-to-2048 sentinel for games that never reach 2048


@dataclass
class EvalWeights:
    """Evaluation weights, tuned by tools/tune_eval.py.

    Defaults: raw tile values weighted by a snake matrix, plus a large bonus per empty
    cell. The scale matters - the empty-cell bonus has to compete with the junk tiles
    the agent would otherwise pile up, otherwise it fills the board and dies.
    """

    corner: float = 1.0  # weight on the position-weighted tile sum
    empty: float = 3000.0  # reward per empty cell
    max_tile: float = 0.0  # reward per log2 of the biggest tile
    smooth: float = 0.0  # penalty per unit of neighbour log2 difference
    mono: float = 0.0  # penalty per monotonicity violation
    tile_score: str = "value"  # 'value' (raw tile), 'log' or 'log2' (log squared)
    row_ratio: float = 16.0  # position weight decay per row away from the corner
    col_ratio: float = 16.0  # position weight decay per column away from the corner
    base: float = 16.0  # position weight of the corner cell
    rotations: int = 1  # 1 = fixed corner, 4 = pick the best corner
    matrix: str = "nneonneo"  # 'gradient', 'snake' or 'nneonneo'
    min_tile: int = 16  # tiles below this value score 0 (small tiles become free)


WEIGHTS = EvalWeights()


def weight_matrix(w: EvalWeights, device=None) -> torch.Tensor:
    """Position weights: base at the corner, decaying by row_ratio/col_ratio.

    `device` keeps the matrix on the same device as the boards it is applied to
    (the teacher runs on the GPU in Colab).
    """
    if w.matrix == "snake":
        m = torch.tensor(
            [[16.0, 8.0, 4.0, 2.0], [2.0, 4.0, 8.0, 16.0], [4.0, 2.0, 1.0, 0.5],
             [2.0, 1.0, 0.5, 0.25]], dtype=torch.float32)
    elif w.matrix == "nneonneo":
        # Classic 2048 potential matrix: 16x decay along a row, 8x down a column.
        m = torch.tensor(
            [[65536.0, 4096.0, 256.0, 16.0], [32768.0, 2048.0, 128.0, 8.0],
             [1024.0, 64.0, 4.0, 4.0], [32.0, 2.0, 2.0, 2.0]], dtype=torch.float32)
    else:
        rows = [w.base * (w.row_ratio ** (3 - i)) / (w.row_ratio ** 3) for i in range(4)]
        cols = [(w.col_ratio ** (3 - j)) / (w.col_ratio ** 3) for j in range(4)]
        m = torch.tensor(rows, dtype=torch.float32)[:, None] * torch.tensor(cols, dtype=torch.float32)[None, :]
    return m.to(device) if device is not None else m


# --------------------------------------------------------------------------- rules


def _compact_left(x: torch.Tensor) -> torch.Tensor:
    nonzero = x != 0
    order = torch.argsort((~nonzero).to(torch.int8), dim=-1, stable=True)
    return torch.gather(x, -1, order)


def _merge_left(x: torch.Tensor) -> torch.Tensor:
    """Slide tiles left and merge equal neighbours, scanning left to right."""
    x = _compact_left(x)
    x0, x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    m01 = (x0 == x1) & (x0 != 0)
    m12 = (x1 == x2) & (x1 != 0) & ~m01
    m23 = (x2 == x3) & (x2 != 0) & ~m12
    o0 = torch.where(m01, 2 * x0, x0)
    o1 = torch.where(m01, torch.where(m23, 2 * x2, x2), torch.where(m12, 2 * x1, x1))
    o2 = torch.where(m01, torch.where(m23, torch.zeros_like(x3), x3),
                     torch.where(m12, x3, torch.where(m23, 2 * x2, x2)))
    o3 = torch.where(m01 | m12 | m23, torch.zeros_like(x3), x3)
    return torch.stack([o0, o1, o2, o3], dim=-1)


def move_boards(B: torch.Tensor, d: int) -> torch.Tensor:
    """B: (M,4,4) int tensor."""
    if d == engine.LEFT:
        return _merge_left(B)
    if d == engine.RIGHT:
        return torch.flip(_merge_left(B.flip(2)), dims=(2,))
    if d == engine.UP:
        return _merge_left(B.transpose(1, 2)).transpose(1, 2)
    if d == engine.DOWN:
        return torch.flip(_merge_left(B.transpose(1, 2).flip(2)).transpose(1, 2), dims=(1,))
    raise ValueError(d)


def spawn_boards(B: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Place one random tile per board: 90% a 2, 10% a 4, in a random empty cell."""
    m = B.shape[0]
    dev = B.device
    flat = B.reshape(m, 16)
    empty = flat == 0
    count = empty.sum(dim=1)
    safe = torch.clamp(count, min=1)
    k = (torch.rand(m, device=dev, generator=gen) * safe).long()
    k = torch.clamp(k, max=safe - 1)
    cum = torch.cumsum(empty.to(torch.int32), dim=1)
    pos = (cum == (k + 1).unsqueeze(1)).to(torch.float32).argmax(dim=1)
    val = torch.where(
        torch.rand(m, device=dev, generator=gen) < engine.SPAWN_P4,
        torch.full((m,), 4, device=dev, dtype=flat.dtype),
        torch.full((m,), 2, device=dev, dtype=flat.dtype),
    )
    out = flat.clone()
    rows = torch.nonzero(count > 0, as_tuple=True)[0]
    out[rows, pos[rows]] = val[rows]
    return out.reshape(B.shape)


def dead_boards(B: torch.Tensor) -> torch.Tensor:
    has_empty = (B == 0).any(dim=1).any(dim=1)
    eq_h = (B[:, :, :-1] == B[:, :, 1:]) & (B[:, :, :-1] != 0)
    eq_v = (B[:, :-1, :] == B[:, 1:, :]) & (B[:, :-1, :] != 0)
    can_merge = eq_h.any(dim=1).any(dim=1) | eq_v.any(dim=1).any(dim=1)
    return ~has_empty & ~can_merge


# ----------------------------------------------------------------------- evaluation


def heuristic(B: torch.Tensor, w: EvalWeights | None = None) -> torch.Tensor:
    """Board quality, higher is better. Hand-tuned, no learning involved."""
    w = w or WEIGHTS
    Bf = B.to(torch.float32)
    logv = torch.where(B > 0, torch.log2(torch.clamp(Bf, min=1.0)), torch.zeros_like(Bf))
    if w.tile_score == "value":
        tile = Bf
    elif w.tile_score == "log2":
        tile = logv * logv
    else:
        tile = logv
    if w.min_tile:
        tile = torch.where(B >= w.min_tile, tile, torch.zeros_like(tile))
    score = w.empty * (B == 0).sum(dim=(1, 2)).to(torch.float32)

    W = weight_matrix(w, B.device)
    best = torch.zeros(B.shape[0], device=B.device, dtype=torch.float32)
    for k in range(w.rotations):
        tilek = torch.rot90(tile, k, dims=(1, 2))
        Wk = torch.rot90(W, k, dims=(0, 1))
        best = torch.maximum(best, (tilek * Wk).sum(dim=(1, 2)))

    mx = torch.log2(torch.clamp(B.amax(dim=(1, 2)), min=1.0))
    score = score + w.corner * best + w.max_tile * mx

    dh = (logv[:, :, :-1] - logv[:, :, 1:]).abs().sum(dim=(1, 2))
    dv = (logv[:, :-1, :] - logv[:, 1:, :]).abs().sum(dim=(1, 2))
    score = score - w.smooth * (dh + dv)

    vh = ((logv[:, :, :-1] > logv[:, :, 1:]) & (B[:, :, :-1] != 0) & (B[:, :, 1:] != 0)).sum(dim=(1, 2))
    vv = ((logv[:, :-1, :] > logv[:, 1:, :]) & (B[:, :-1, :] != 0) & (B[:, 1:, :] != 0)).sum(dim=(1, 2))
    score = score - w.mono * (vh + vv).to(torch.float32)

    # A dead board keeps its potential: it is the game-over position, not a hole in
    # the search. (Zeroing it makes the endgame blind - every leaf of a full board
    # would read 0 and the search could not see that merging frees space.)
    return score


# ---------------------------------------------------------------------------- search


@torch.no_grad()
def expectimax(boards: torch.Tensor, depth: int, cell_samples: int = 16,
               gen: torch.Generator | None = None, w: EvalWeights | None = None):
    """Expectimax over a batch of boards with (optionally sampled) chance nodes.

    Each level: every board plays each of the 4 moves; for each move we pick
    `cell_samples` empty cells (16 = all of them) and place both a 2 and a 4 in each,
    so a chance node averages over the real spawn distribution instead of a handful of
    random draws.

    Returns (values, best_moves, root_scores).
    """
    dev = boards.device
    root_n = boards.shape[0]
    cur = boards
    levels = []
    level_dead = []

    for _ in range(depth):
        n = cur.shape[0]
        dead_mask = dead_boards(cur)
        pot_dead = heuristic(cur[dead_mask], w) if bool(dead_mask.any()) else None
        level_dead.append((dead_mask, pot_dead))
        moved = [move_boards(cur, d) for d in range(4)]
        changed = [((moved[d] != cur).any(dim=(1, 2))) for d in range(4)]
        pair_board = torch.cat(moved, dim=0)  # (4n,4,4): pair p = board p%n, move p//n
        pair_owner = torch.arange(npairs := pair_board.shape[0], device=dev) % n
        pair_changed = torch.cat(changed, dim=0)

        flat = pair_board.reshape(npairs, 16)
        empty = flat == 0
        if cell_samples >= 16:
            keys = torch.where(empty, torch.zeros_like(flat, dtype=torch.float32),
                               torch.full_like(flat, float("inf"), dtype=torch.float32))
        else:
            rand = torch.rand(npairs, 16, device=dev, generator=gen)
            keys = torch.where(empty, rand, torch.full_like(rand, float("inf")))
        order = torch.argsort(keys, dim=1)[:, :cell_samples]
        ok = torch.isfinite(keys.gather(1, order))
        cells = order

        base = pair_board.repeat_interleave(cell_samples, dim=0)
        child = base.repeat_interleave(2, dim=0)
        nchild = child.shape[0]
        flatc = child.reshape(nchild, 16)
        rows = torch.arange(nchild, device=dev)
        flatc[rows, cells.reshape(-1).repeat_interleave(2)] = torch.where(
            (torch.arange(nchild, device=dev) % 2) == 0,
            torch.full((nchild,), 2, device=dev, dtype=child.dtype),
            torch.full((nchild,), 4, device=dev, dtype=child.dtype))
        child_pair = torch.arange(npairs, device=dev).repeat_interleave(2 * cell_samples)
        child_ok = ok.reshape(-1).repeat_interleave(2)

        keep = child_ok
        child = child[keep]
        child_pair = child_pair[keep]
        levels.append((pair_owner, pair_changed, child_pair, n))
        cur = child

    vals = heuristic(cur, w)
    root_scores = None
    for lvl in range(depth - 1, -1, -1):
        pair_owner, pair_changed, child_pair, nboards = levels[lvl]
        npairs = pair_owner.shape[0]
        sums = torch.zeros(npairs, device=dev, dtype=vals.dtype).scatter_add_(0, child_pair, vals)
        counts = torch.zeros(npairs, device=dev, dtype=vals.dtype).scatter_add_(
            0, child_pair, torch.ones_like(vals))
        pair_val = sums / counts.clamp(min=1)
        pair_val = torch.where(pair_changed, pair_val, torch.full_like(pair_val, INVALID))
        board_val = torch.full((nboards,), INVALID, device=dev, dtype=vals.dtype).scatter_reduce_(
            0, pair_owner, pair_val, reduce="amax")
        # a board with no legal move is game over: score it by its potential
        dead_mask, pot_dead = level_dead[lvl]
        if pot_dead is not None:
            full_pot = torch.zeros(nboards, device=dev, dtype=vals.dtype)
            full_pot[dead_mask] = pot_dead
            board_val = torch.where(dead_mask, full_pot, board_val)
        if lvl == 0:
            root_scores = pair_val.view(4, root_n).t()  # pairs are ordered (move, board)
        vals = board_val

    best_moves = root_scores.argmax(dim=1)
    return vals, best_moves, root_scores
