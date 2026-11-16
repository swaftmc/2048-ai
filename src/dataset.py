"""Teacher games -> labelled dataset.

Each teacher game contributes one sample per move: the board it played from, the move
the teacher chose, and how many moves were left before a 2048 tile appeared.
"""
from __future__ import annotations

import numpy as np
import torch

from src import engine, teacher


def generate_dataset(games: int = 100, depth: int = 3, cell_samples: int = 16, batch: int = 8,
                     seed: int = 0, max_moves: int = 4000, device: str = "cpu",
                     fail_label: float = teacher.FAIL_LABEL, weights: teacher.EvalWeights | None = None,
                     verbose: bool = False):
    """Run the teacher on `games` games and return labelled states.

    cell_samples=16 enumerates every empty cell (exact chance nodes); lower values
    sample that many cells per chance node, which is cheaper but noisier.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    w = weights or teacher.WEIGHTS

    boards_out, moves_out, values_out, ends_out, teacher_vals = [], [], [], [], []
    stats = {"games": 0, "reached_2048": 0, "moves_to_2048": [], "max_tile": [], "states": 0}
    done = 0

    while done < games:
        b = min(batch, games - done)
        boards = torch.zeros((b, 4, 4), dtype=torch.int32, device=device)
        boards = teacher.spawn_boards(boards, gen)
        boards = teacher.spawn_boards(boards, gen)

        records = [[] for _ in range(b)]  # (board, move) per step
        reached_at = [None] * b
        ended_at = [None] * b
        alive = np.ones(b, dtype=bool)

        for step in range(max_moves):
            idx = np.nonzero(alive)[0]
            if idx.size == 0:
                break
            cur = boards[torch.as_tensor(idx, device=device)]
            values, moves, _ = teacher.expectimax(cur, depth, cell_samples, gen, w)
            moves_np = moves.cpu().numpy()
            cur_np = cur.cpu().numpy().astype(np.int16)
            vals_np = values.cpu().numpy().astype(np.float32)
            for j, gi in enumerate(idx):
                records[gi].append((cur_np[j], int(moves_np[j]), float(vals_np[j])))

            for d in range(4):
                sel = idx[moves_np == d]
                if sel.size:
                    sel_t = torch.as_tensor(sel, device=device)
                    boards[sel_t] = teacher.move_boards(boards[sel_t], d)
            boards = teacher.spawn_boards(boards, gen)

            got = (boards == teacher.TARGET_TILE).any(dim=(1, 2)).cpu().numpy()
            for gi in idx[got[idx]]:
                reached_at[gi] = step + 1
                ended_at[gi] = step + 1
                alive[gi] = False
            dead = teacher.dead_boards(boards).cpu().numpy()
            for gi in idx[dead[idx]]:
                ended_at[gi] = step + 1
                alive[gi] = False

        for gi in range(b):
            steps = reached_at[gi]
            end = ended_at[gi] if ended_at[gi] is not None else len(records[gi])
            for t, (board, move, tval) in enumerate(records[gi]):
                boards_out.append(board)
                moves_out.append(move)
                values_out.append(float(steps - t) if steps is not None else fail_label)
                ends_out.append(float(end - t))
                teacher_vals.append(float(tval))
            stats["games"] += 1
            stats["states"] += len(records[gi])
            if steps is not None:
                stats["reached_2048"] += 1
                stats["moves_to_2048"].append(steps)
            stats["max_tile"].append(int(boards[gi].max().item()))
        done += b

        if verbose:
            reached = stats["reached_2048"]
            mean_moves = float(np.mean(stats["moves_to_2048"])) if stats["moves_to_2048"] else 0.0
            print(f"[teacher] games={stats['games']}/{games} states={stats['states']} "
                  f"reached2048={reached} mean_moves={mean_moves:.1f}", flush=True)

    return {
        "boards": np.stack(boards_out).astype(np.int16),
        "move": np.asarray(moves_out, dtype=np.int8),
        "value": np.asarray(values_out, dtype=np.float32),
        "value_to_end": np.asarray(ends_out, dtype=np.float32),
        "teacher_value": np.asarray(teacher_vals, dtype=np.float32),
        "stats": stats,
    }
