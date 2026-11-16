"""The student: two linear heads over board features, plus the agent that plays with them.

- policy head: multinomial logistic regression -> which move the teacher would play.
  Trained as "which of the 4 moves", so the accuracy metric is top-1 agreement with
  the depth-4 teacher.
- value head: ridge regression -> expected moves remaining until a 2048 tile appears.
  The browser agent searches over this value (expectimax), which is cheap because a
  linear evaluation is one dot product.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from src import engine, features as feat

MODEL_FORMAT = "2048-linear-v1"
INVALID = -1.0e7


@dataclass
class Model:
    policy_w: np.ndarray  # (4, F)
    policy_b: np.ndarray  # (4,)
    value_w: np.ndarray  # (F,)  search head: lower = a better position
    value_b: float
    value_target: str = "eval"  # 'eval' (teacher's static evaluation) or 'moves_to_2048'
    moves_w: np.ndarray | None = None  # reporting head: moves remaining to a 2048 tile
    moves_b: float = 0.0
    moves_mean: float = 0.0
    moves_std: float = 1.0
    feature_mean: np.ndarray | None = None  # feature standardisation
    feature_std: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    # -- inference ---------------------------------------------------------------
    def transform(self, phi: np.ndarray) -> np.ndarray:
        """Standardise the features (the heads were fitted on standardised inputs)."""
        phi = np.asarray(phi, dtype=np.float64)
        if self.feature_mean is None or self.feature_std is None:
            return phi
        return (phi - self.feature_mean) / self.feature_std

    def policy_scores(self, phi: np.ndarray) -> np.ndarray:
        return self.transform(phi) @ self.policy_w.T + self.policy_b

    def value_raw(self, phi: np.ndarray) -> np.ndarray:
        """Linear output of the value head (search uses this; only the order matters)."""
        return self.transform(phi) @ self.value_w + self.value_b

    def value(self, phi: np.ndarray) -> np.ndarray:
        """Predicted moves remaining until a 2048 tile appears (lower is better)."""
        if self.moves_w is None:
            return np.zeros(len(np.atleast_1d(phi)))
        raw = self.transform(phi) @ self.moves_w + self.moves_b
        return np.maximum(raw * self.moves_std + self.moves_mean, 0.0)

    # -- io ----------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format": MODEL_FORMAT,
            "n_features": int(self.policy_w.shape[1]),
            "policy": {
                "coef": [[float(x) for x in row] for row in self.policy_w],
                "intercept": [float(x) for x in self.policy_b],
            },
            "value": {
                "coef": [float(x) for x in self.value_w],
                "intercept": float(self.value_b),
                "target": self.value_target,
            },
            "value_moves": None if self.moves_w is None else {
                "coef": [float(x) for x in self.moves_w],
                "intercept": float(self.moves_b),
                "mean": float(self.moves_mean),
                "std": float(self.moves_std),
            },
            "scaler": {
                "mean": [float(x) for x in (self.feature_mean if self.feature_mean is not None
                                            else np.zeros(self.policy_w.shape[1]))],
                "std": [float(x) for x in (self.feature_std if self.feature_std is not None
                                           else np.ones(self.policy_w.shape[1]))],
            },
            "meta": self.meta,
        }

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)

    @staticmethod
    def from_dict(d: dict) -> "Model":
        if d.get("format") != MODEL_FORMAT:
            raise ValueError(f"unexpected model format {d.get('format')!r}")
        val = d["value"]
        scaler = d.get("scaler", {})
        n_feat = int(d["n_features"])
        moves = d.get("value_moves")
        return Model(
            policy_w=np.asarray(d["policy"]["coef"], dtype=np.float64),
            policy_b=np.asarray(d["policy"]["intercept"], dtype=np.float64),
            value_w=np.asarray(val["coef"], dtype=np.float64),
            value_b=float(val["intercept"]),
            value_target=str(val.get("target", "moves_to_2048")),
            moves_w=None if moves is None else np.asarray(moves["coef"], dtype=np.float64),
            moves_b=0.0 if moves is None else float(moves["intercept"]),
            moves_mean=0.0 if moves is None else float(moves.get("mean", 0.0)),
            moves_std=1.0 if moves is None else float(moves.get("std", 1.0)),
            feature_mean=np.asarray(scaler.get("mean", np.zeros(n_feat)), dtype=np.float64),
            feature_std=np.asarray(scaler.get("std", np.ones(n_feat)), dtype=np.float64),
            meta=d.get("meta", {}),
        )

    @staticmethod
    def load(path: str) -> "Model":
        with open(path, encoding="utf-8") as fh:
            return Model.from_dict(json.load(fh))


# --------------------------------------------------------------------------- fitting


def fit_policy(X: np.ndarray, y: np.ndarray, C: float = 100.0):
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(X, y)
    return clf


def fit_value(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> Ridge:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X, y)
    return ridge


def build_model(clf, ridge, meta: dict, value_target: str = "eval",
                moves_ridge=None, moves_mean: float = 0.0, moves_std: float = 1.0,
                feature_mean: np.ndarray | None = None,
                feature_std: np.ndarray | None = None) -> Model:
    return Model(
        policy_w=np.asarray(clf.coef_, dtype=np.float64),
        policy_b=np.asarray(clf.intercept_, dtype=np.float64),
        value_w=np.asarray(ridge.coef_, dtype=np.float64).reshape(-1),
        value_b=float(np.asarray(ridge.intercept_).reshape(-1)[0]),
        value_target=value_target,
        moves_w=None if moves_ridge is None else np.asarray(moves_ridge.coef_,
                                                           dtype=np.float64).reshape(-1),
        moves_b=0.0 if moves_ridge is None else float(
            np.asarray(moves_ridge.intercept_).reshape(-1)[0]),
        moves_mean=float(moves_mean),
        moves_std=float(moves_std),
        feature_mean=None if feature_mean is None else np.asarray(feature_mean, dtype=np.float64),
        feature_std=None if feature_std is None else np.asarray(feature_std, dtype=np.float64),
        meta=meta,
    )


# ----------------------------------------------------------------------------- agent


def legal_moves(board: np.ndarray) -> list[int]:
    return [d for d in range(4) if not np.array_equal(engine.move_boards(board[None], d)[0], board)]


def policy_action(board: np.ndarray, model: Model) -> int:
    """Pure ML: argmax of the policy head, ignoring moves that do nothing."""
    scores = model.policy_scores(feat.features(board))
    legal = legal_moves(board)
    if not legal:
        return -1
    best = legal[0]
    for d in legal[1:]:
        if scores[d] > scores[best]:
            best = d
    return best


def _proposals(n: int, k: int, rng: np.random.Generator, moved: np.ndarray):
    """(cells, values, valid) for k spawn proposals per node."""
    if k >= 32:  # exact enumeration of every (cell, value) pair
        cells = np.broadcast_to(np.tile(np.repeat(np.arange(16), 2), 2)[:k][None, :], (n, k))
        vals = np.broadcast_to(np.tile(np.array([2, 4]), 16)[:k][None, :], (n, k))
    else:
        cells = rng.integers(0, 16, size=(n, k))
        vals = np.where(rng.random((n, k)) < engine.SPAWN_P4, 4, 2)
    occupied = np.take_along_axis(moved.reshape(n, 16), cells, axis=1)
    return cells, vals, occupied == 0


def _search_values(nodes: np.ndarray, model: Model, depth: int, k: int, rng: np.random.Generator):
    """Expectimax values for a batch of nodes (higher is better).

    The search runs on the value head's raw output (lower = fewer moves to 2048, or a
    lower teacher evaluation), so no clipping is needed and the ordering matches the
    browser exactly.
    """
    n = nodes.shape[0]
    if depth == 0:
        return -model.value_raw(feat.features(nodes))
    per_move = []
    for d in range(4):
        moved = engine.move_boards(nodes, d)
        valid_move = (moved != nodes).any(axis=(1, 2))
        cells, vals, ok = _proposals(n, k, rng, moved)
        child = np.repeat(moved[:, None, :], k, axis=1).copy()
        flat = child.reshape(n, k, 16)
        flat[np.arange(n)[:, None], np.arange(k)[None, :], cells] = vals
        v = _search_values(child.reshape(n * k, 4, 4), model, depth - 1, k, rng).reshape(n, k)
        v = np.where(ok, v, 0.0)
        count = ok.sum(axis=1)
        mean = np.where(count > 0, v.sum(axis=1) / np.maximum(count, 1), INVALID)
        mean = np.where(valid_move & np.isfinite(mean), mean, INVALID)
        per_move.append(mean)
    return np.maximum.reduce(per_move)


def search_action(board: np.ndarray, model: Model, depth: int = 2, k: int = 32, seed: int = 0) -> int:
    """Expectimax over the linear value model; returns the chosen move."""
    rng = np.random.default_rng(seed)
    nodes = board[None]
    moved = [engine.move_boards(nodes, d)[0] for d in range(4)]
    legal = [d for d in range(4) if not np.array_equal(moved[d], board)]
    if not legal:
        return -1
    if depth <= 0:
        scores = -model.value_raw(feat.features(np.stack(moved)))
        scores = np.array([scores[d] if d in legal else INVALID for d in range(4)])
        return int(np.argmax(scores))
    values = []
    for d in range(4):
        if d not in legal:
            values.append(INVALID)
            continue
        cells, vals, ok = _proposals(1, k, rng, moved[d][None])
        child = np.repeat(moved[d][None, None, :, :], k, axis=1).copy()
        flat = child.reshape(1, k, 16)
        flat[0, np.arange(k), cells[0]] = vals[0]
        v = _search_values(child.reshape(k, 4, 4), model, depth - 1, k, rng).reshape(1, k)
        v = np.where(ok, v, 0.0)
        count = ok.sum(axis=1)
        mean = v.sum(axis=1) / np.maximum(count, 1) if count[0] > 0 else np.array([INVALID])
        values.append(mean[0] if np.isfinite(mean[0]) else INVALID)
    scores = np.asarray(values)
    return int(np.argmax(scores))


def play_game(model: Model, mode: str = "search", depth: int = 2, k: int = 32, seed: int = 0,
              max_moves: int = 4000):
    """Play one game. mode is 'policy' or 'search'."""
    rng = np.random.default_rng(seed)
    board = engine.new_board(rng)
    moves = 0
    while True:
        if mode == "policy":
            d = policy_action(board, model)
        else:
            d = search_action(board, model, depth=depth, k=k, seed=seed * 7919 + moves)
        if d < 0:
            break
        new = engine.move_boards(board[None], d)[0]
        if np.array_equal(new, board):
            break
        board = engine.spawn_boards(new[None], rng)[0]
        moves += 1
        if engine.has_tile(board[None], 2048)[0]:
            return {"reached_2048": True, "moves_to_2048": moves, "max_tile": int(board.max()),
                    "moves": moves}
        if engine.dead_boards(board[None])[0] or moves >= max_moves:
            break
    return {"reached_2048": False, "moves_to_2048": None, "max_tile": int(board.max()),
            "moves": moves}


def evaluate(model: Model, mode: str = "search", games: int = 20, depth: int = 2, k: int = 32,
             seed: int = 0, verbose: bool = False) -> dict:
    results = []
    for i in range(games):
        results.append(play_game(model, mode=mode, depth=depth, k=k, seed=seed + i))
        if verbose:
            r = results[-1]
            print(f"  [{mode} d{depth}] game {i + 1}/{games}: "
                  f"max tile {r['max_tile']}, "
                  f"{'2048 in ' + str(r['moves_to_2048']) + ' moves' if r['reached_2048'] else 'no 2048'}",
                  flush=True)
    wins = [r for r in results if r["reached_2048"]]
    return {
        "mode": mode,
        "depth": depth,
        "games": games,
        "success_rate": len(wins) / max(games, 1),
        "mean_moves_to_2048": (float(np.mean([r["moves_to_2048"] for r in wins]))
                               if wins else None),
        "median_moves_to_2048": (float(np.median([r["moves_to_2048"] for r in wins]))
                                 if wins else None),
        "mean_max_tile": float(np.mean([r["max_tile"] for r in results])),
        "max_tile_histogram": {str(t): int(c) for t, c in
                               zip(*np.unique([r["max_tile"] for r in results],
                                              return_counts=True))},
    }
