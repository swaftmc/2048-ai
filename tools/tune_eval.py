"""Grid-search the teacher's evaluation weights.

    python -m tools.tune_eval --games 6 --depth 2

Scores each candidate by how often the teacher reaches a 2048 tile and how many moves
that took. The winning weights are the defaults in src/teacher.py.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import torch

from src import teacher


def play(w: teacher.EvalWeights, games: int, depth: int, samples: int, seed: int,
         max_moves: int = 1200) -> dict:
    gen = torch.Generator()
    gen.manual_seed(seed)
    B = torch.zeros((games, 4, 4), dtype=torch.int32)
    B = teacher.spawn_boards(B, gen)
    B = teacher.spawn_boards(B, gen)
    alive = np.ones(games, dtype=bool)
    reached = np.zeros(games, dtype=int)
    for step in range(max_moves):
        idx = np.nonzero(alive)[0]
        if idx.size == 0:
            break
        cur = B[torch.as_tensor(idx)]
        _, moves, _ = teacher.expectimax(cur, depth, samples, gen, w)
        moves_np = moves.cpu().numpy()
        for d in range(4):
            sel = idx[moves_np == d]
            if sel.size:
                sel_t = torch.as_tensor(sel)
                before = B[sel_t].clone()
                B[sel_t] = teacher.move_boards(B[sel_t], d)
                stuck = (B[sel_t] == before).all(dim=(1, 2)).cpu().numpy()
                alive[sel[stuck]] = False
        B = teacher.spawn_boards(B, gen)
        got = (B == teacher.TARGET_TILE).any(dim=(1, 2)).cpu().numpy()
        for gi in idx[got[idx]]:
            reached[gi] = step + 1
            alive[gi] = False
        dead = teacher.dead_boards(B).cpu().numpy()
        alive[idx[dead[idx]]] = False
    ok = reached > 0
    return {
        "success": float(ok.mean()),
        "mean_moves": float(reached[ok].mean()) if ok.any() else None,
        "mean_max": float(B.amax(dim=(1, 2)).max().item()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", type=int, default=6)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

def _v(empty: float = 0.0, rotations: int = 1, matrix: str = "nneonneo", smooth: float = 0.0,
      mono: float = 0.0, base: float = 16.0, ratio: float = 16.0, min_tile: int = 0,
      tile_score: str = "value", max_tile: float = 0.0) -> teacher.EvalWeights:
    return teacher.EvalWeights(tile_score=tile_score, empty=empty, smooth=smooth, mono=mono,
                                row_ratio=ratio, col_ratio=ratio, base=base, rotations=rotations,
                                matrix=matrix, min_tile=min_tile, max_tile=max_tile)


CANDIDATES = [
    ("base min16", _v(empty=3000, min_tile=16)),
    ("base min8", _v(empty=3000, min_tile=8)),
    ("base min4", _v(empty=3000, min_tile=4)),
    ("base min0", _v(empty=3000, min_tile=0)),
    ("min16 mt200", _v(empty=3000, min_tile=16, max_tile=200)),
    ("min16 mt2000", _v(empty=3000, min_tile=16, max_tile=2000)),
    ("min8 mt2000", _v(empty=3000, min_tile=8, max_tile=2000)),
    ("min4 mt2000", _v(empty=3000, min_tile=4, max_tile=2000)),
    ("min16 e1000 mt2000", _v(empty=1000, min_tile=16, max_tile=2000)),
    ("min16 e10000 mt2000", _v(empty=10000, min_tile=16, max_tile=2000)),
    ("min16 s50", _v(empty=3000, min_tile=16, smooth=50)),
    ("min16 m50", _v(empty=3000, min_tile=16, mono=50)),
]


def random_candidates(n: int, seed: int = 0):
    """Random search over the evaluation parameter space (depth 1 is fast enough)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        matrix = rng.choice(["snake", "gradient", "nneonneo"])
        ratio = float(rng.choice([2.0, 4.0, 8.0, 16.0]))
        out.append((
            f"rnd {matrix} ratio={ratio} empty={rng.choice([100,300,1000,3000,10000])}",
            teacher.EvalWeights(
                tile_score=str(rng.choice(["value", "log", "log2"])),
                empty=float(rng.choice([100.0, 300.0, 1000.0, 3000.0, 10000.0])),
                smooth=float(rng.choice([0.0, 50.0, 200.0])),
                mono=float(rng.choice([0.0, 50.0, 200.0])),
                max_tile=float(rng.choice([0.0, 50.0, 200.0])),
                row_ratio=ratio, col_ratio=ratio,
                rotations=int(rng.choice([1, 4])),
                matrix=matrix,
            )))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", type=int, default=4)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-moves", type=int, default=800)
    p.add_argument("--match", type=str, default=None, help="only run candidates containing this")
    p.add_argument("--random", type=int, default=0, help="random search over N configs")
    args = p.parse_args()

    candidates = CANDIDATES if not args.random else random_candidates(args.random, args.seed)
    best = None
    for label, w in candidates:
        if args.match and args.match not in label:
            continue
        res = play(w, args.games, args.depth, args.samples, args.seed, args.max_moves)
        score = (res["success"], -(res["mean_moves"] or 1e6))
        print(f"{res['success']:.2f} moves={res['mean_moves']} max={res['mean_max']:.0f}  {label}",
              flush=True)
        if best is None or score > best[0]:
            best = (score, label, w, res)
    print("\nBEST:", best[1], best[2], best[3])


if __name__ == "__main__":
    main()
