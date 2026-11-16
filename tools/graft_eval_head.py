"""Graft a freshly fitted evaluation head onto an existing model.

The teacher's *static* evaluation is a fixed function of the board, not something that
depends on which games produced it. So the head that the browser's search needs can be
fitted on any set of boards - and grafted onto a policy head that was trained on a
bigger teacher run (e.g. a Colab one) without redoing that run.

    python tools/graft_eval_head.py --model model.json --dataset dataset.npz --out model_grafted.json

The feature scaler of the target model is reused, so the grafted head is fitted in
exactly the space the model does inference in.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import features as feat, student  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="model.json")
    p.add_argument("--dataset", default="dataset.npz")
    p.add_argument("--out", default=None)
    p.add_argument("--alpha", type=float, default=1.0)
    args = p.parse_args()

    model_path = pathlib.Path(args.model)
    data = dict(json.loads(model_path.read_text(encoding="utf-8")))
    model = student.Model.from_dict(data)

    with np.load(args.dataset, allow_pickle=True) as z:
        boards = z["boards"].astype(np.int64)
        target = z["eval_value"].astype(np.float64) if "eval_value" in z else None
    if target is None:
        # older caches predate the field: the evaluation is a function of the board, so
        # recompute it instead of re-running the teacher
        from src import teacher
        import torch

        print(f"{args.dataset} has no eval_value - recomputing it from the boards")
        target = teacher.heuristic(torch.as_tensor(boards).int(), teacher.WEIGHTS).numpy()
        target = target.astype(np.float64)
    print(f"fitting the evaluation head on {len(boards)} states from {args.dataset}")

    X = feat.features(boards)
    # fit in the target model's scaled space, so inference stays consistent
    Xs = model.transform(X)
    y = -target
    ridge = student.fit_value(Xs, y, alpha=args.alpha)
    resid = ridge.predict(Xs) - y
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    scale = float(np.std(y))
    print(f"evaluation head: rmse {rmse:.4g} on a target spread of {scale:.4g} "
          f"({100 * rmse / scale:.3f}% of the signal)")

    # keep the old value head as the reporting head if it was one
    old = data.get("value", {})
    if data.get("value_moves") is None and "mean" in old and "std" in old:
        data["value_moves"] = {
            "coef": old["coef"], "intercept": old["intercept"],
            "mean": old["mean"], "std": old["std"],
        }
        print("kept the previous value head as the moves-to-2048 reporting head")

    data["value"] = {
        "coef": [float(x) for x in np.asarray(ridge.coef_).reshape(-1)],
        "intercept": float(np.asarray(ridge.intercept_).reshape(-1)[0]),
        "target": "eval",
    }
    out = pathlib.Path(args.out) if args.out else model_path
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
