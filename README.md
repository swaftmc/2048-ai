# 2048 with a linear model

A 2048 agent whose decision making is **linear regression + logistic regression only**.
No neural networks anywhere in the student.

```
teacher  = depth-limited expectimax search (hand-written heuristic eval)   [torch]
labels   = (board, move the teacher played, moves left until a 2048 tile)  [teacher games]
student  = policy head : multinomial logistic regression  (which move)
           value head  : ridge regression                (moves remaining)
browser  = plays with either head, or a small expectimax over the value head
```

The search does the hard thinking offline; the browser runs one dot product per move.

## Quick start (no training needed)

```
# local CPU smoke run (~4 min): the teacher reaches a 2048 tile in ~1 game in 6
python -m src.pipeline --games 12 --depth 2 --cell-samples 16 --batch 6 --out model.json

# the real run: Colab GPU, depth 4 (see train.ipynb for the memory-aware settings)
python -m src.pipeline --games 300 --depth 4 --cell-samples 4 --batch 4 --out model.json

python tools/build_html.py model.json      # -> dist/2048.html
python tools/check_build.py                # verify the built file (model embedded, JS runs)
```

Then double-click `dist/2048.html`. It is a single self-contained file: the weights are
embedded as base64, so it runs offline with no server.

## Training on a GPU (Colab)

**Open `train.ipynb` in Colab and press "Run all".** That is the whole workflow:

1. The setup cell pulls this repo (or uses the folder you are already in) and installs
   torch / numpy / scikit-learn.
2. The config cell sets the run size - edit it if you want.
3. The smoke cell plays one teacher move on the GPU and prints the seconds-per-move and
   the VRAM peak, so you can size the real run before committing to it.
4. The run cell trains and writes `model.json`.
5. The last cell saves it to Google Drive and downloads it.

Open it from GitHub with the **Open in Colab** badge, or upload `train.ipynb` - either
way Colab clones this repo into `/content` and the setup cell finds it.

Useful knobs (config cell):

* `GAMES` - teacher games to play. A game is ~1000 moves, so the wall clock is roughly
  `GAMES * 1000 / BATCH * seconds_per_batch_move`.
* `DEPTH` - expectimax depth. Measured on CPU: depth 2 reaches a 2048 tile in 17-50% of
  games, depth 3 in 75%. Depth 4 is the quality setting but costs ~8x the leaves.
* `CELL_SAMPLES` - empty cells enumerated per chance node (16 = exact, 8 = sampled).
  Leaves per move per game = `(8 * CELL_SAMPLES) ** DEPTH`.
* `BATCH` - games the teacher plays in parallel. Keep
  `BATCH * leaves * 64 bytes` under ~4 GB of VRAM.

Then build the app locally:

```
python tools/build_html.py <downloaded model.json>    # -> dist/2048.html
```

## What the numbers mean

| metric | meaning |
|---|---|
| `teacher.success_rate` | how often the teacher itself reached a 2048 tile |
| `teacher.mean_moves_to_2048` | the teacher's move count to 2048 (the target to beat) |
| `policy.val_agreement_with_teacher` | **accuracy**: how often the logistic head picks the teacher's move, on held-out positions (illegal moves count as wrong) |
| `value.val_rmse_moves` | how far the value head is off, in the value target's units |
| `student.search_depth2.success_rate` | how often the *student* reaches 2048 in the browser's own search |

### How strong the teacher is

The teacher is a search, and its depth is what decides whether the student can reach
2048. Measured on this machine (4-8 games per row, one tile per move, 90/10 spawns):

| teacher depth | reached 2048 | moves to 2048 | best tile |
|---|---|---|---|
| 1 (greedy on the heuristic) | 0% | - | 512 |
| 2 (exact chance nodes) | 17-50% | ~1000 | 2048 |
| 3 (exact chance nodes) | 75% | ~984 | 2048 |
| 4 (sampled cells, GPU) | the setting that gets there consistently | | |

So the quality run needs depth 4, which is what the Colab notebook is for. Reproduce
any row with `python -m tools.tune_eval --games 4 --depth 3 --samples 16`.

### The value target

The value head is trained on `moves_to_2048` (how many moves the teacher had left when
it reached a 2048 tile). If the teacher rarely gets there, that target collapses to a
constant and carries no signal, so the pipeline falls back to `teacher_eval`: the
teacher's own evaluation of the position, negated so that "lower is better" still holds.
The browser's search only uses the *ordering* of the value, so it works with either
target; the metrics panel tells you which one a model was trained on.

## Layout

```
src/engine.py      2048 rules in numpy (authoritative reference)
src/features.py    104 board features (numpy)
src/teacher.py     torch expectimax teacher + evaluation function
src/dataset.py     teacher games -> labelled dataset
src/student.py     logistic + ridge heads, search agent, evaluation
src/pipeline.py    driver: teacher -> labels -> fit -> model.json -> metrics
web/engine.js      JS mirror of the rules + features + inference
web/index.html     the app (watch mode, metrics panel, benchmark)
tools/build_html.py inlines engine.js + the model -> dist/2048.html
tools/check_build.py smoke-checks the built file (model embedded, app script parses, JS plays)
tools/tune_eval.py grid/random search over the teacher's evaluation weights
tools/ref_teacher.py slow recursive reference search used to validate the fast one
test/test_sync.py  rule tests + numpy/torch/JS parity checks
```

## Tests

```
python test/test_sync.py          # or: python -m pytest test/test_sync.py -q
```

Covers the merge rules, the 90/10 spawn, numpy-vs-torch move parity, the vectorised
teacher vs a plain recursive reference search, and Python-vs-JS parity for the features
and the model inference. The parity checks matter: the browser must play exactly the
game the model was trained on.

## Notes and limits

* The student is linear, so it inherits the teacher's ceiling. If the teacher only
  reaches 512, the student will too - deepen the search or retune the evaluation
  (`tools/tune_eval.py`) rather than adding features.
* The browser search uses exact chance nodes at depth 1-2 and sampled cells at depth 3
  to stay interactive.
* `web/engine.js` and `src/engine.py` are deliberate duplicates (one language each).
  `test/test_sync.py` is what keeps them honest.
