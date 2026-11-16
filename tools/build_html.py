"""Inline a trained model into the web app and emit a single self-contained HTML file.

    python tools/build_html.py model.json
    -> dist/2048.html   (double-click to play, no server, no network)
"""
from __future__ import annotations

import base64
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PLACEHOLDER = "__MODEL_B64__"


def build(model_path: str, out_path: str | None = None) -> str:
    model_path = pathlib.Path(model_path)
    with open(model_path, "rb") as fh:
        raw = fh.read()
    # sanity: it must be the format the web app expects
    data = json.loads(raw.decode("utf-8"))
    if data.get("format") != "2048-linear-v1":
        raise SystemExit(f"unexpected model format: {data.get('format')!r}")
    encoded = base64.b64encode(raw).decode("ascii")

    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    if PLACEHOLDER not in html:
        raise SystemExit(f"placeholder {PLACEHOLDER} not found in web/index.html")
    html = html.replace(PLACEHOLDER, encoded)

    engine_js = (ROOT / "web" / "engine.js").read_text(encoding="utf-8")
    tag = '<script src="engine.js"></script>'
    if tag not in html:
        raise SystemExit("expected <script src=\"engine.js\"></script> in web/index.html")
    html = html.replace(tag, "<script>\n" + engine_js + "\n</script>")

    out = pathlib.Path(out_path) if out_path else ROOT / "dist" / "2048.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    size = out.stat().st_size / 1024
    print(f"wrote {out} ({size:.0f} KB, model {len(encoded) / 1024:.0f} KB base64)")
    return str(out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python tools/build_html.py <model.json> [out.html]")
    build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
