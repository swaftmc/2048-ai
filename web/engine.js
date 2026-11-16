/* 2048 rules, board features and linear-model inference.
 *
 * This is the JS mirror of src/engine.py and src/features.py. test/test_sync.py
 * asserts the two implementations agree to 1e-9, so the browser plays exactly the
 * game the Python pipeline was trained on.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Engine = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const LOG_MAX = 11;
  const W_CORNER = [[16, 8, 4, 2], [8, 4, 2, 1], [4, 2, 1, 0.5], [2, 1, 0.5, 0.25]];
  // The position weights the teacher itself uses (see src/features.py).
  const W_NNEONNE = [[65536, 4096, 256, 16], [32768, 2048, 128, 8], [1024, 64, 4, 4],
                     [32, 2, 2, 2]];
  const BIG_TILE = 16;
  const EVAL_SCALE = 1e6;
  const N_FEATURES = 104;

  /* --------------------------------------------------------------- game rules */
  function compactLeft(row) {
    const out = row.filter(v => v !== 0);
    while (out.length < 4) out.push(0);
    return out;
  }
  function mergeLeft(row) {
    const x = compactLeft(row);
    const [x0, x1, x2, x3] = x;
    const m01 = x0 === x1 && x0 !== 0;
    const m12 = !m01 && x1 === x2 && x1 !== 0;
    const m23 = !m12 && x2 === x3 && x2 !== 0;
    if (m01) {
      return (x2 === x3 && x2 !== 0) ? [2 * x0, 2 * x2, 0, 0] : [2 * x0, x2, x3, 0];
    }
    if (m12) return [x0, 2 * x1, x3, 0];
    if (m23) return [x0, x1, 2 * x2, 0];
    return [x0, x1, x2, x3];
  }
  const transpose = b => [0, 1, 2, 3].map(j => [0, 1, 2, 3].map(i => b[i][j]));
  function moveBoard(b, d) {
    if (d === 0) return b.map(mergeLeft);
    if (d === 1) return b.map(r => mergeLeft(r.slice().reverse()).reverse());
    const t = transpose(b);
    return transpose(d === 2 ? t.map(mergeLeft)
                             : t.map(r => mergeLeft(r.slice().reverse()).reverse()));
  }
  function sameBoard(a, b) {
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) if (a[i][j] !== b[i][j]) return false;
    return true;
  }
  function emptyCells(b) {
    const out = [];
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) if (b[i][j] === 0) out.push([i, j]);
    return out;
  }
  function isDead(b) {
    if (emptyCells(b).length) return false;
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      if (j < 3 && b[i][j] === b[i][j + 1] && b[i][j] !== 0) return false;
      if (i < 3 && b[i][j] === b[i + 1][j] && b[i][j] !== 0) return false;
    }
    return true;
  }
  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function spawn(b, rng) {
    const cells = emptyCells(b);
    if (!cells.length) return b;
    const c = cells[Math.floor(rng() * cells.length)];
    const out = b.map(r => r.slice());
    out[c[0]][c[1]] = rng() < 0.1 ? 4 : 2;
    return out;
  }

  /* ------------------------------------------------------------ board features */
  function rot90(a) { return transpose(a).reverse(); }
  function rotN(a, k) { let r = a; for (let i = 0; i < k; i++) r = rot90(r); return r; }

  function features(b) {
    const f = [];
    const logv = b.map(row => row.map(v => (v > 0 ? Math.log2(v) : 0)));
    const mx = Math.max(...b.flat());
    for (const row of logv) for (const v of row) f.push(v / LOG_MAX);
    for (const row of b) for (const v of row) f.push(v === 0 ? 1 : 0);
    for (const row of b) for (const v of row) f.push(v === mx && v > 0 ? 1 : 0);
    for (let i = 0; i < 4; i++) f.push(logv[i].reduce((a, v) => a + v, 0) / (4 * LOG_MAX));
    for (let j = 0; j < 4; j++) {
      let s = 0; for (let i = 0; i < 4; i++) s += logv[i][j];
      f.push(s / (4 * LOG_MAX));
    }
    f.push(Math.log2(Math.max(mx, 1)) / LOG_MAX);
    f.push(emptyCells(b).length / 16);
    let sm = 0, np = 0;
    for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) {
      if (b[i][j] !== 0 && b[i][j + 1] !== 0) { sm += Math.abs(logv[i][j] - logv[i][j + 1]); np++; }
    }
    for (let i = 0; i < 3; i++) for (let j = 0; j < 4; j++) {
      if (b[i][j] !== 0 && b[i + 1][j] !== 0) { sm += Math.abs(logv[i][j] - logv[i + 1][j]); np++; }
    }
    f.push(sm / Math.max(np, 1) / LOG_MAX);
    let mono = 0;
    for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++)
      if (b[i][j] !== 0 && b[i][j + 1] !== 0 && logv[i][j] > logv[i][j + 1]) mono++;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 4; j++)
      if (b[i][j] !== 0 && b[i + 1][j] !== 0 && logv[i][j] > logv[i + 1][j]) mono++;
    f.push(mono / 24);
    let mg = 0;
    for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++)
      if (b[i][j] !== 0 && b[i][j] === b[i][j + 1]) mg++;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 4; j++)
      if (b[i][j] !== 0 && b[i][j] === b[i + 1][j]) mg++;
    f.push(mg / 24);
    for (const t of [32, 64, 128, 256]) {
      let c = 0; for (const row of b) for (const v of row) if (v >= t) c++;
      f.push(c / 16);
    }
    const corners = [b[0][0], b[0][3], b[3][0], b[3][3]];
    f.push(corners.some(v => v === mx && mx > 0) ? 1 : 0);
    let onEdge = false;
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      const edge = (i === 0 || i === 3 || j === 0 || j === 3);
      const corner = (i === 0 || i === 3) && (j === 0 || j === 3);
      if (edge && !corner && b[i][j] === mx && mx > 0) onEdge = true;
    }
    f.push(onEdge ? 1 : 0);
    let bestGrad = 0;
    for (let k = 0; k < 4; k++) {
      const lk = rotN(logv, k), Wk = rotN(W_CORNER, k);
      let s = 0;
      for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) s += lk[i][j] * Wk[i][j];
      f.push(s / (LOG_MAX * 16));
      const rows = [0, 1, 2, 3].map(i => lk[i].reduce((a, v) => a + v, 0));
      let g = 0;
      for (let i = 0; i < 3; i++) g += Math.max(rows[i] - rows[i + 1], 0);
      bestGrad = Math.max(bestGrad, g / (4 * LOG_MAX));
    }
    f.push(bestGrad);
    // teacher-evaluation mirrors: raw tile value x the teacher's position weights
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      f.push(b[i][j] * W_NNEONNE[i][j] / EVAL_SCALE);
    }
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      const v = b[i][j] >= BIG_TILE ? b[i][j] : 0;
      f.push(v * W_NNEONNE[i][j] / EVAL_SCALE);
    }
    if (f.length !== N_FEATURES) throw new Error("feature count " + f.length);
    return f;
  }

  /* ----------------------------------------------------------------- inference */
  function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function valueRaw(model, phi) {
    return dot(model.value.coef, transform(model, phi)) + model.value.intercept;
  }
  function valueMoves(model, phi) {
    const v = model.value;
    const mean = v.mean === undefined ? 0 : v.mean;
    const std = v.std === undefined ? 1 : v.std;
    return Math.max(Math.expm1(valueRaw(model, phi) * std + mean), 0);
  }

  function policyScores(model, phi) {
    const t = transform(model, phi);
    return model.policy.coef.map((c, d) => dot(c, t) + model.policy.intercept[d]);
  }
  function transform(model, phi) {
    const s = model.scaler;
    if (!s || !s.mean || !s.std) return phi;
    const out = new Array(phi.length);
    for (let i = 0; i < phi.length; i++) {
      const sd = s.std[i] ? s.std[i] : 1;
      out[i] = (phi[i] - s.mean[i]) / sd;
    }
    return out;
  }
  /* Expectimax over the linear value model. Exact chance nodes (every empty cell x
     {2,4}); deeper searches sample cells so the browser stays responsive. */
  function searchValue(model, board, depth, k, rng) {
    if (depth === 0) return -valueRaw(model, features(board));
    let best = -Infinity;
    for (let d = 0; d < 4; d++) {
      const moved = moveBoard(board, d);
      if (sameBoard(moved, board)) continue;
      const cells = emptyCells(moved);
      if (!cells.length) continue;
      let sum = 0, count = 0;
      if (k >= 32) {
        for (const [i, j] of cells) for (const v of [2, 4]) {
          const b2 = moved.map(r => r.slice()); b2[i][j] = v;
          sum += searchValue(model, b2, depth - 1, k, rng); count++;
        }
      } else {
        for (let s = 0; s < k; s++) {
          const ci = Math.floor(rng() * 16), i = ci >> 2, j = ci & 3;
          if (moved[i][j] !== 0) continue;
          const v = rng() < 0.1 ? 4 : 2;
          const b2 = moved.map(r => r.slice()); b2[i][j] = v;
          sum += searchValue(model, b2, depth - 1, k, rng); count++;
        }
        if (!count) continue;
      }
      best = Math.max(best, sum / count);
    }
    return best;
  }

  function chooseMove(model, board, mode, depth, rng) {
    if (mode === "policy") {
      const sc = policyScores(model, features(board));
      let best = -1, bestScore = -Infinity;
      for (let d = 0; d < 4; d++) {
        if (sameBoard(moveBoard(board, d), board)) continue;
        if (sc[d] > bestScore) { bestScore = sc[d]; best = d; }
      }
      return best;
    }
    const k = depth >= 3 ? 8 : 32;
    let best = -1, bestScore = -Infinity;
    for (let d = 0; d < 4; d++) {
      const moved = moveBoard(board, d);
      if (sameBoard(moved, board)) continue;
      let sum = 0, count = 0;
      if (k >= 32) {
        for (const [i, j] of emptyCells(moved)) for (const v of [2, 4]) {
          const b2 = moved.map(r => r.slice()); b2[i][j] = v;
          sum += searchValue(model, b2, depth - 1, k, rng); count++;
        }
      } else {
        for (let s = 0; s < k; s++) {
          const ci = Math.floor(rng() * 16), i = ci >> 2, j = ci & 3;
          if (moved[i][j] !== 0) continue;
          const v = rng() < 0.1 ? 4 : 2;
          const b2 = moved.map(r => r.slice()); b2[i][j] = v;
          sum += searchValue(model, b2, depth - 1, k, rng); count++;
        }
      }
      if (!count) continue;
      const val = sum / count;
      if (val > bestScore) { bestScore = val; best = d; }
    }
    return best;
  }

  return {
    LOG_MAX, N_FEATURES, W_CORNER,
    compactLeft, mergeLeft, transpose, moveBoard, sameBoard, emptyCells, isDead, spawn,
    mulberry32, features, rot90, rotN, dot, policyScores, valueRaw, valueMoves, searchValue,
    chooseMove,
  };
});
