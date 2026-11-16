"""Training pipeline: teacher games -> linear heads -> model.json -> metrics.

    python -m src.pipeline --games 200 --depth 4 --samples 4 --out model.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from src import dataset, engine, features as feat, student


def teacher_agreement(model: student.Model, boards: np.ndarray, teacher_move: np.ndarray) -> float:
    """Top-1 agreement with the teacher, counting illegal (no-op) picks as wrong."""
    legal = engine.changed_moves(boards.astype(np.int64))
    scores = model.policy_scores(feat.features(boards.astype(np.int64)))
    scores = np.where(legal, scores, -np.inf)
    return float((scores.argmax(axis=1) == teacher_move).mean())


def run(games: int = 200, depth: int = 3, cell_samples: int = 16, batch: int = 8, seed: int = 0,
        out: str = "model.json", eval_games: int = 20, val_frac: float = 0.2,
        device: str | None = None, quiet: bool = False) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    data = dataset.generate_dataset(games=games, depth=depth, cell_samples=cell_samples, batch=batch,
                                    seed=seed, device=device, verbose=not quiet)
    boards = data["boards"].astype(np.int64)
    X = feat.features(boards)
    y_move = data["move"].astype(np.int64)

    # Value target: "moves remaining until a 2048 tile appears". If the teacher rarely
    # gets there that target is a constant sentinel and carries no signal, so fall back
    # to the teacher's own evaluation of the position (negated, so that lower stays
    # better and the browser's search keeps the same meaning).
    success_rate = data["stats"]["reached_2048"] / max(data["stats"]["games"], 1)
    if success_rate >= 0.25:
        y_value_raw = data["value"].astype(np.float64)
        value_target = "moves_to_2048"
    else:
        y_value_raw = -data["teacher_value"].astype(np.float64)
        value_target = "teacher_eval"
    # standardise so the ridge penalty means the same thing for either target
    v_mean = float(y_value_raw.mean())
    v_std = float(y_value_raw.std()) or 1.0
    y_value = (y_value_raw - v_mean) / v_std

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_val = max(1, int(len(X) * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # standardise the features: they live on wildly different scales, and both heads
    # are penalised by their regularisation
    f_mean = X[train_idx].mean(axis=0)
    f_std = X[train_idx].std(axis=0)
    f_std[f_std < 1e-9] = 1.0
    Xs = (X - f_mean) / f_std

    clf = student.fit_policy(Xs[train_idx], y_move[train_idx])
    ridge = student.fit_value(Xs[train_idx], y_value[train_idx])

    meta = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "teacher": {"depth": depth, "cell_samples": cell_samples, "games": games, "seed": seed},
        "states": int(len(X)),
        "train_states": int(len(train_idx)),
        "value_target": value_target,
    }
    model = student.build_model(clf, ridge, meta, value_mean=v_mean, value_std=v_std,
                                feature_mean=f_mean, feature_std=f_std)

    pred_raw = model.value_raw(X[val_idx]) * v_std + v_mean
    true_moves = y_value_raw[val_idx]
    metrics = {
        "teacher": {
            "games": data["stats"]["games"],
            "states": data["stats"]["states"],
            "reached_2048": data["stats"]["reached_2048"],
            "success_rate": data["stats"]["reached_2048"] / max(data["stats"]["games"], 1),
            "mean_moves_to_2048": (float(np.mean(data["stats"]["moves_to_2048"]))
                                   if data["stats"]["moves_to_2048"] else None),
            "mean_max_tile": float(np.mean(data["stats"]["max_tile"])),
        },
        "policy": {
            "val_agreement_with_teacher": teacher_agreement(model, boards[val_idx], y_move[val_idx]),
            "train_accuracy": float(clf.score(Xs[train_idx], y_move[train_idx])),
        },
        "value": {
            "val_mae_moves": float(np.mean(np.abs(pred_raw - true_moves))),
            "val_rmse_moves": float(np.sqrt(np.mean((pred_raw - true_moves) ** 2))),
            "units": "moves to 2048" if value_target == "moves_to_2048" else "teacher eval units",
        },
    }

    if eval_games > 0:
        metrics["student"] = {
            "policy_only": student.evaluate(model, mode="policy", games=eval_games, seed=seed + 1),
            "search_depth1": student.evaluate(model, mode="search", depth=1, games=eval_games,
                                              seed=seed + 2),
            "search_depth2": student.evaluate(model, mode="search", depth=2, games=eval_games,
                                              seed=seed + 3),
        }

    metrics["elapsed_sec"] = round(time.time() - t0, 1)
    model.meta["metrics"] = metrics
    model.to_json(out)
    with open(out.replace(".json", "_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    if not quiet:
        print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--cell-samples", type=int, default=16,
                   help="empty cells enumerated per chance node (16 = exact)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="model.json")
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    run(games=args.games, depth=args.depth, cell_samples=args.cell_samples, batch=args.batch,
        seed=args.seed, out=args.out, eval_games=args.eval_games, device=args.device)


if __name__ == "__main__":
    main()
