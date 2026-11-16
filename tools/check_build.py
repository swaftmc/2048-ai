"""Smoke-check the built single-file app: the model is embedded and the JS engine works.

    python tools/check_build.py [dist/2048.html]
"""
from __future__ import annotations

import base64
import json
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> None:
    html_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist" / "2048.html"
    html = html_path.read_text(encoding="utf-8")
    assert "__MODEL_B64__" not in html, "model placeholder was not replaced"
    assert "function moveBoard" in html, "engine.js was not inlined"

    m = re.search(r'const MODEL_B64 = "([A-Za-z0-9+/=]+)";', html)
    assert m, "could not find the embedded model"
    model = json.loads(base64.b64decode(m.group(1)).decode("utf-8"))
    assert model["format"] == "2048-linear-v1"
    print(f"embedded model: {model['n_features']} features, "
          f"value target {model['meta'].get('value_target')}")

    # syntax-check the app script that is inlined in the same file
    blocks = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(blocks) >= 2, "expected the engine and the app script to be inlined"
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as fh:
        fh.write(blocks[-1])
        tmp_js = fh.name
    try:
        proc = subprocess.run(["node", "--check", tmp_js], capture_output=True, text=True)
        assert proc.returncode == 0, f"app script has syntax errors: {proc.stderr}"
    finally:
        pathlib.Path(tmp_js).unlink(missing_ok=True)
    print("app script: syntax ok")

    # play a few games with the JS engine + the embedded model
    script = """
    const fs = require('fs');
    const Engine = require('./web/engine.js');
    const model = JSON.parse(fs.readFileSync(process.argv[process.argv.length - 1], 'utf8'));
    let wins = 0, best = 0, moves = [];
    for (let g = 0; g < 3; g++) {
      const rng = Engine.mulberry32(g + 1);
      let b = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
      b = Engine.spawn(b, rng); b = Engine.spawn(b, rng);
      let mv = 0, hit = null;
      while (mv < 800) {
        const d = Engine.chooseMove(model, b, 'search', 2, Engine.mulberry32(g * 991 + mv));
        if (d < 0) break;
        const moved = Engine.moveBoard(b, d);
        if (Engine.sameBoard(moved, b)) break;
        b = Engine.spawn(moved, Engine.mulberry32(g * 7717 + mv));
        mv++;
        if (Math.max(...b.flat()) >= 2048) { hit = mv; break; }
        if (Engine.isDead(b)) break;
      }
      best = Math.max(best, Math.max(...b.flat()));
      if (hit) { wins++; moves.push(hit); }
    }
    console.log(JSON.stringify({wins, best, moves}));
    """
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(model, fh)
        tmp = fh.name
    try:
        proc = subprocess.run(["node", "-e", script, tmp], capture_output=True, text=True,
                              cwd=str(ROOT), timeout=300)
        if proc.returncode != 0:
            raise SystemExit(f"node failed: {proc.stderr}")
        out = json.loads(proc.stdout)
    finally:
        pathlib.Path(tmp).unlink(missing_ok=True)
    print(f"3 games in the JS engine: {out['wins']} reached 2048, best tile {out['best']}, "
          f"moves {out['moves']}")


if __name__ == "__main__":
    main()
