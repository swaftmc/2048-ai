"""Cross-checks between the Python and JS implementations, plus rule sanity tests.

    python -m pytest test/test_sync.py -q      (or: python test/test_sync.py)

The Python/JS parity checks matter: the browser plays with the same rules and the same
features the model was trained on, so any drift would silently ruin the agent.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import engine, features as feat, student, teacher  # noqa: E402
from tools import ref_teacher  # noqa: E402


def random_board(rng: np.random.Generator, max_tiles: int = 12, max_exp: int = 9) -> np.ndarray:
    b = np.zeros((4, 4), dtype=np.int64)
    for _ in range(rng.integers(0, max_tiles)):
        i, j = rng.integers(0, 4, 2)
        b[i, j] = 2 ** rng.integers(1, max_exp)
    return b


# --------------------------------------------------------------------- game rules


def test_merge_rules():
    b = np.array([[2, 2, 4, 4], [8, 0, 16, 0], [2, 2, 2, 2], [0, 4, 0, 4]])
    assert engine.move_boards(b[None], 0)[0].tolist() == [
        [4, 8, 0, 0], [8, 16, 0, 0], [4, 4, 0, 0], [8, 0, 0, 0]]
    assert engine.move_boards(b[None], 1)[0].tolist() == [
        [0, 0, 4, 8], [0, 0, 8, 16], [0, 0, 4, 4], [0, 0, 0, 8]]
    # a tile merges at most once per move
    assert engine.move_boards(np.array([[[2, 2, 2, 2]]]), 0)[0].tolist() == [[4, 4, 0, 0]]
    # the scan is left to right, so an equal pair that straddles the (0,1)/(2,3)
    # boundary still merges: [2,8,8,4] -> [2,16,4,0]
    assert engine.move_boards(np.array([[[2, 8, 8, 4]]]), 0)[0].tolist() == [[2, 16, 4, 0]]
    assert engine.move_boards(np.array([[[2, 4, 4, 8]]]), 0)[0].tolist() == [[2, 8, 8, 0]]
    assert engine.move_boards(np.array([[[4, 4, 8, 8]]]), 0)[0].tolist() == [[8, 16, 0, 0]]
    # vertical moves
    col = np.array([[2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert engine.move_boards(col[None], 2)[0].tolist() == [[4, 0, 0, 0], [0, 0, 0, 0],
                                                            [0, 0, 0, 0], [0, 0, 0, 0]]
    assert engine.move_boards(col[None], 3)[0].tolist() == [[0, 0, 0, 0], [0, 0, 0, 0],
                                                            [0, 0, 0, 0], [4, 0, 0, 0]]


def test_spawn_distribution():
    rng = np.random.default_rng(0)
    b = np.zeros((4000, 4, 4), dtype=np.int64)
    s = engine.spawn_boards(b, rng)
    assert (s > 0).sum() == 4000
    p4 = float((s == 4).sum()) / 4000
    assert 0.08 < p4 < 0.12, p4


def test_dead_detection():
    full = np.array([[2, 4, 8, 16], [4, 8, 16, 32], [8, 16, 32, 64], [16, 32, 64, 2]])
    assert engine.dead_boards(full[None])[0]
    mergeable = full.copy()
    mergeable[0, 0] = 4
    assert not engine.dead_boards(mergeable[None])[0]
    assert not engine.dead_boards(np.zeros((1, 4, 4), dtype=np.int64))[0]


# ------------------------------------------------------------------ numpy vs torch


def test_torch_engine_matches_numpy():
    rng = np.random.default_rng(1)
    for _ in range(200):
        b = random_board(rng)
        bt = torch.as_tensor(b[None]).int()
        for d in range(4):
            assert np.array_equal(engine.move_boards(b[None], d)[0],
                                  teacher.move_boards(bt, d)[0].numpy())


def test_teacher_matches_reference_search():
    """The vectorised torch search must pick the same moves as the plain recursive one."""
    w = teacher.EvalWeights()
    rng = np.random.default_rng(2)
    gen = torch.Generator()
    gen.manual_seed(2)
    for _ in range(10):
        b = random_board(rng)
        _, tm, _ = teacher.expectimax(torch.as_tensor(b[None]).int(), 2, 16, gen, w)
        _, rm = ref_teacher.expectimax(b, 2, w)
        assert int(tm[0]) == rm, (b.tolist(), int(tm[0]), rm)


# --------------------------------------------------------------- python vs JS


def _run_node(script: str, payload: dict) -> dict:
    tmp = ROOT / "test" / "_tmp_payload.json"
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    try:
        proc = subprocess.run(
            ["node", "-e", script, str(tmp)],
            capture_output=True, text=True, cwd=str(ROOT), timeout=120,
        )
        if proc.returncode != 0:
            raise AssertionError(f"node failed: {proc.stderr}")
        return json.loads(proc.stdout)
    finally:
        tmp.unlink(missing_ok=True)


NODE_SCRIPT = """
const fs = require('fs');
const Engine = require('./web/engine.js');
const payload = JSON.parse(fs.readFileSync(process.argv[process.argv.length - 1], 'utf8'));
const out = payload.boards.map(b => Engine.features(b));
console.log(JSON.stringify(out));
"""


def test_features_match_js():
    rng = np.random.default_rng(3)
    boards = [random_board(rng) for _ in range(60)]
    js = _run_node(NODE_SCRIPT, {"boards": [b.tolist() for b in boards]})
    py = feat.features(np.stack(boards))
    assert py.shape == (60, feat.N_FEATURES)
    for i in range(60):
        assert np.allclose(py[i], np.asarray(js[i]), atol=1e-9), i


NODE_MOVE_SCRIPT = """
const fs = require('fs');
const Engine = require('./web/engine.js');
const payload = JSON.parse(fs.readFileSync(process.argv[process.argv.length - 1], 'utf8'));
const out = payload.boards.map(b => [0,1,2,3].map(d => Engine.moveBoard(b, d)));
console.log(JSON.stringify(out));
"""


def test_moves_match_js():
    rng = np.random.default_rng(4)
    boards = [random_board(rng) for _ in range(60)]
    js = _run_node(NODE_MOVE_SCRIPT, {"boards": [b.tolist() for b in boards]})
    for i, b in enumerate(boards):
        for d in range(4):
            assert np.array_equal(engine.move_boards(b[None], d)[0], np.asarray(js[i][d])), (i, d)


NODE_INFER_SCRIPT = """
const fs = require('fs');
const Engine = require('./web/engine.js');
const payload = JSON.parse(fs.readFileSync(process.argv[process.argv.length - 1], 'utf8'));
const model = payload.model;
const out = payload.boards.map(b => {
  const phi = Engine.features(b);
  return {policy: Engine.policyScores(model, phi), value: Engine.valueMoves(model, phi)};
});
console.log(JSON.stringify(out));
"""


def test_inference_matches_js():
    """The JS inference must reproduce the Python one for a real model."""
    rng = np.random.default_rng(5)
    boards = [random_board(rng) for _ in range(40)]
    X = feat.features(np.stack(boards))
    policy = rng.normal(size=(4, feat.N_FEATURES)) * 0.1
    value = rng.normal(size=feat.N_FEATURES) * 0.1
    model = student.Model(policy_w=policy, policy_b=np.zeros(4), value_w=value, value_b=0.5,
                          value_target="eval",
                          moves_w=value, moves_b=0.5, moves_mean=10.0, moves_std=3.0)
    payload = {
        "boards": [b.tolist() for b in boards],
        "model": model.to_dict(),
    }
    js = _run_node(NODE_INFER_SCRIPT, payload)
    for i in range(40):
        py_policy = model.policy_scores(X[i])
        py_value = model.value(X[i])
        assert np.allclose(py_policy, np.asarray(js[i]["policy"]), atol=1e-8), i
        assert abs(float(py_value) - float(js[i]["value"])) < 1e-8, i


# --------------------------------------------------------------------- student


def test_student_fit_and_roundtrip(tmp_path=None):
    rng = np.random.default_rng(6)
    boards = [random_board(rng) for _ in range(400)]
    X = feat.features(np.stack(boards))
    y_move = rng.integers(0, 4, 400)
    y_value = rng.random(400) * 200
    clf = student.fit_policy(X, y_move)
    ridge = student.fit_value(X, np.log1p(y_value))
    model = student.build_model(clf, ridge, {"test": True})
    path = str(ROOT / "test" / "_tmp_model.json")
    model.to_json(path)
    loaded = student.Model.load(path)
    (ROOT / "test" / "_tmp_model.json").unlink(missing_ok=True)
    assert np.allclose(loaded.policy_w, model.policy_w)
    assert np.allclose(loaded.value(X[:5]), model.value(X[:5]))
    assert student.policy_action(boards[0], model) in range(4)


NODE_CHOOSE_SCRIPT = """
const fs = require('fs');
const Engine = require('./web/engine.js');
const payload = JSON.parse(fs.readFileSync(process.argv[process.argv.length - 1], 'utf8'));
const out = payload.boards.map(b =>
  Engine.chooseMove(payload.model, b, 'search', 1, Engine.mulberry32(1)));
console.log(JSON.stringify(out));
"""


def test_search_action_matches_js():
    """Depth-1 search is deterministic, so both implementations must pick the same move."""
    rng = np.random.default_rng(7)
    boards = [random_board(rng, max_tiles=8) for _ in range(25)]
    X = feat.features(np.stack(boards))
    model = student.Model(
        policy_w=rng.normal(size=(4, feat.N_FEATURES)) * 0.1,
        policy_b=np.zeros(4),
        value_w=rng.normal(size=feat.N_FEATURES) * 0.1,
        value_b=0.5,
        value_target="eval",
        moves_w=rng.normal(size=feat.N_FEATURES) * 0.1,
        moves_b=0.5,
        moves_mean=10.0,
        moves_std=3.0,
    )
    js = _run_node(NODE_CHOOSE_SCRIPT, {"boards": [b.tolist() for b in boards],
                                        "model": model.to_dict()})
    for i, b in enumerate(boards):
        py = student.search_action(b, model, depth=1, k=32, seed=1)
        assert int(js[i]) == py, (i, int(js[i]), py, b.tolist())


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
