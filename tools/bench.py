"""Benchmark the model with the browser's own engine (node) or with the Python agent.

    python tools/bench.py model.json [--games 20] [--depth 2] [--engine js|python]

The JS numbers are what the shipped HTML will do, so they are the ones that matter.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NODE_BENCH = """
const fs = require('fs');
const Engine = require('./web/engine.js');
const model = JSON.parse(fs.readFileSync(process.argv[1], 'utf8'));
const games = parseInt(process.argv[2], 10);
const depth = parseInt(process.argv[3], 10);
const mode = process.argv[4];
let wins = 0, moves = [], best = 0, tiles = [];
for (let g = 0; g < games; g++) {
  const rng = Engine.mulberry32(g + 1);
  let b = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
  b = Engine.spawn(b, rng); b = Engine.spawn(b, rng);
  let mv = 0, hit = null;
  while (mv < 3000) {
    const d = Engine.chooseMove(model, b, mode, depth, Engine.mulberry32(g * 7919 + mv));
    if (d < 0) break;
    const moved = Engine.moveBoard(b, d);
    if (Engine.sameBoard(moved, b)) break;
    b = Engine.spawn(moved, Engine.mulberry32(g * 104729 + mv));
    mv++;
    if (Math.max(...b.flat()) >= 2048) { hit = mv; break; }
    if (Engine.isDead(b)) break;
  }
  const mx = Math.max(...b.flat());
  best = Math.max(best, mx);
  tiles.push(mx);
  if (hit) { wins++; moves.push(hit); }
  console.log(`  game ${g + 1}/${games}: ${hit ? '2048 in ' + hit + ' moves' : 'max tile ' + mx}`);
}
const mean = moves.length ? moves.reduce((a, b) => a + b, 0) / moves.length : null;
console.log(JSON.stringify({wins, games, mean_moves: mean, best, tiles}));
"""


def bench_js(model_path: pathlib.Path, games: int, depth: int, mode: str) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        fh.write(model_path.read_text(encoding="utf-8"))
        tmp = fh.name
    try:
        proc = subprocess.run(["node", "-e", NODE_BENCH, tmp, str(games), str(depth), mode],
                              capture_output=True, text=True, cwd=str(ROOT), timeout=7200)
        if proc.returncode != 0:
            raise SystemExit(f"node failed: {proc.stderr[-2000:]}")
        lines = proc.stdout.strip().splitlines()
        return json.loads(lines[-1])
    finally:
        pathlib.Path(tmp).unlink(missing_ok=True)


def bench_python(model_path: pathlib.Path, games: int, depth: int) -> dict:
    from src import student

    model = student.Model.load(str(model_path))
    res = student.evaluate(model, mode="search", depth=depth, games=games, verbose=True)
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", nargs="?", default="model.json")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--engine", choices=["js", "python"], default="js")
    p.add_argument("--mode", choices=["search", "policy"], default="search")
    args = p.parse_args()

    model_path = pathlib.Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"{model_path} not found")
    meta = json.loads(model_path.read_text(encoding="utf-8")).get("meta", {})
    print(f"model: teacher {meta.get('teacher')} | {meta.get('states')} states | "
          f"target {meta.get('value_target')}")

    if args.engine == "js":
        print(f"JS engine, {args.mode} mode, depth {args.depth}, {args.games} games:")
        res = bench_js(model_path, args.games, args.depth, args.mode)
        print(f"\n{res['wins']}/{res['games']} reached 2048"
              + (f", mean {res['mean_moves']:.0f} moves" if res.get("mean_moves") else "")
              + f", best tile {res['best']}")
        print("tiles:", res["tiles"])
    else:
        res = bench_python(model_path, args.games, args.depth)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
