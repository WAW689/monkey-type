/* 浏览器内置的三个实验（Web Worker）。
 * 与 monkey/exp1_random.py / exp2_weasel.py / exp3_rl.py 的算法一致，
 * 只是随机数换成可复现的 mulberry32，并且用分组 softmax 加速 REINFORCE。
 */
'use strict';

const DEFAULT_TARGET = '生存还是毁灭，这是一个问题';
let TARGET = DEFAULT_TARGET;
const BOS = '^', EOS = '$';

// ---------------------------------------------------------------- 随机数
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function buildVocab(corpus, target) {
  const set = new Set();
  for (const ch of target) set.add(ch);
  for (const line of corpus) for (const ch of line) set.add(ch);
  return [...set].sort();
}

function decode(codes, off, len, vocab) {
  let s = '';
  for (let i = 0; i < len; i++) s += vocab[codes[off + i]];
  return s;
}

function matchCount(s, target) {
  let n = 0;
  for (let i = 0; i < target.length; i++) if (s[i] === target[i]) n++;
  return n;
}

function distinct2(samples) {
  const seen = new Set();
  let total = 0;
  for (const s of samples) {
    for (let i = 0; i + 1 < s.length; i++) { seen.add(s.slice(i, i + 2)); total++; }
  }
  return seen.size / Math.max(1, total);
}

function thin(arr, max) {
  if (arr.length <= max) return arr;
  const out = [];
  for (let i = 0; i < max; i++) out.push(arr[Math.round(i * (arr.length - 1) / (max - 1))]);
  return out;
}

// ---------------------------------------------------------------- 字符三元语言模型
class CharTrigramLM {
  constructor(corpus, vocab) {
    this.vocab = vocab;
    this.V = vocab.length;
    this.S = vocab.length + 2;       // + BOS + EOS
    this.BOS = vocab.length;
    this.EOS = vocab.length + 1;
    this.ids = new Map(vocab.map((c, i) => [c, i]));
    this.l3 = 0.5; this.l2 = 0.3; this.l1 = 0.15; this.l0 = 0.05;

    const S = this.S;
    this.uni = new Int32Array(S);
    this.ctx1 = new Int32Array(S);
    this.bi = new Map();
    this.tri = new Map();
    this.ctx2 = new Map();
    this.N = 0;
    for (const line of corpus) {
      const seq = [this.BOS, this.BOS];
      for (const ch of line) seq.push(this.ids.get(ch));
      seq.push(this.EOS);
      for (let i = 2; i < seq.length; i++) {
        const a = seq[i - 2], b = seq[i - 1], c = seq[i];
        this.uni[c]++;
        const bk = b * S + c;
        this.bi.set(bk, (this.bi.get(bk) || 0) + 1);
        this.ctx1[b]++;
        const tk = (a * S + b) * S + c;
        this.tri.set(tk, (this.tri.get(tk) || 0) + 1);
        const ck = a * S + b;
        this.ctx2.set(ck, (this.ctx2.get(ck) || 0) + 1);
        this.N++;
      }
    }
  }

  prob(a, b, c) {
    const S = this.S;
    const n2 = this.ctx2.get(a * S + b) || 0;
    const p3 = n2 ? (this.tri.get((a * S + b) * S + c) || 0) / n2 : 0;
    const n1 = this.ctx1[b] || 0;
    const p2 = n1 ? (this.bi.get(b * S + c) || 0) / n1 : 0;
    const p1 = this.uni[c] / this.N;
    return this.l3 * p3 + this.l2 * p2 + this.l1 * p1 + this.l0 / (this.V + 1);
  }

  logprob(s) {
    const ids = [];
    for (const ch of s) {
      const id = this.ids.get(ch);
      if (id === undefined) return -Infinity;
      ids.push(id);
    }
    let lp = 0;
    let a = this.BOS, b = this.BOS;
    for (const c of ids) { lp += Math.log(this.prob(a, b, c)); a = b; b = c; }
    lp += Math.log(this.prob(a, b, this.EOS));
    return lp;
  }

  fluency(s) {
    let len = 0; for (const _ of s) len++;
    return this.logprob(s) / (len + 1);
  }

  perplexity(s) { return Math.exp(-this.fluency(s)); }
}

// ---------------------------------------------------------------- 实验①：纯随机
function theory(V, L) {
  const log10Space = L * Math.log10(V);
  const universe = 4.35e17, speed = 1e9;
  const log10Seconds = log10Space - Math.log10(speed);
  return {
    vocab_size: V, length: L, log10_search_space: log10Space,
    attempts_per_second: speed, log10_expected_seconds: log10Seconds,
    log10_universe_ages: log10Seconds - Math.log10(universe),
  };
}

function runExp1(vocab, rand) {
  const V = vocab.length, L = TARGET.length;
  const tgt = new Int16Array(L);
  for (let i = 0; i < L; i++) tgt[i] = vocab.indexOf(TARGET[i]);
  const hist = new Int32Array(L + 1);
  const buf = new Int16Array(L);
  const history = [], samples = [];
  let best = -1, bestStr = '';
  const attempts = 4_000_000, batch = 200_000, logEvery = 20_000;

  for (let done = 0; done < attempts; done += batch) {
    for (let k = 0; k < batch; k++) {
      let m = 0;
      for (let i = 0; i < L; i++) {
        const c = (rand() * V) | 0;
        buf[i] = c;
        if (c === tgt[i]) m++;
      }
      hist[m]++;
      if (m > best) { best = m; bestStr = decode(buf, 0, L, vocab); }
      if ((done + k + 1) % logEvery === 0) {
        history.push({ attempt: done + k + 1, best, sample: decode(buf, 0, L, vocab) });
      }
      if (samples.length < 300) samples.push(decode(buf, 0, L, vocab));
    }
  }
  return {
    name: '纯随机猴子', attempts, best_match: best, best_string: bestStr,
    match_histogram: Array.from(hist), history, samples, theory: theory(V, L),
  };
}

// ---------------------------------------------------------------- 实验②：黄鼠狼程序
function runExp2(vocab, rand) {
  const V = vocab.length, L = TARGET.length;
  const tgt = new Int16Array(L);
  for (let i = 0; i < L; i++) tgt[i] = vocab.indexOf(TARGET[i]);
  const pop = 100, mu = 0.04, maxGen = 20_000;
  let parent = new Int16Array(L);
  for (let i = 0; i < L; i++) parent[i] = (rand() * V) | 0;

  const history = [];
  let gen = 0, hit = false;
  for (let g = 0; g <= maxGen; g++) {
    gen = g;
    let fit = 0;
    for (let i = 0; i < L; i++) if (parent[i] === tgt[i]) fit++;
    history.push({ gen, best: decode(parent, 0, L, vocab), fitness: fit });
    if (fit === L) { hit = true; break; }

    let bestFit = fit, bestKid = parent;
    for (let k = 0; k < pop; k++) {
      const kid = parent.slice();
      for (let i = 0; i < L; i++) {
        if (rand() < mu) kid[i] = (rand() * V) | 0;
      }
      let f = 0;
      for (let i = 0; i < L; i++) if (kid[i] === tgt[i]) f++;
      if (f > bestFit) { bestFit = f; bestKid = kid; if (f === L) break; }
    }
    parent = bestKid;
  }
  return {
    name: '黄鼠狼进化', pop, mu, hit,
    generations: gen, evaluations: gen * pop, history: thin(history, 400),
  };
}

// ---------------------------------------------------------------- 实验③：REINFORCE
function runExp3(vocab, lm, mode, seed, onProgress) {
  const V = vocab.length, L = TARGET.length, BOSI = V;
  // 默认目标 300 步；目标更长时适当加步数（最多 600 步）
  const steps = Math.min(600, 300 + Math.max(0, L - 13) * 25);
  const batch = 64, lr = 0.05, wTarget = 0.6, entropy = 0.01, logEvery = 1;
  const uniform = mode === 'sparse'; // 稀疏奖励几乎恒为 0，策略保持均匀，可直接均匀采样
  const rand = mulberry32(seed);
  const tgt = new Int16Array(L);
  for (let i = 0; i < L; i++) tgt[i] = vocab.indexOf(TARGET[i]);

  const Wp = new Float32Array(L * V);
  const Wb = new Float32Array((V + 1) * V);
  const mWp = new Float32Array(L * V), vWp = new Float32Array(L * V);
  const mWb = new Float32Array((V + 1) * V), vWb = new Float32Array((V + 1) * V);
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;
  let adamT = 0;

  function adam(param, off, grad, m, v) {
    const c1 = 1 - Math.pow(b1, adamT), c2 = 1 - Math.pow(b2, adamT);
    for (let i = 0; i < V; i++) {
      const g = grad[i], j = off + i;
      m[j] = b1 * m[j] + (1 - b1) * g;
      v[j] = b2 * v[j] + (1 - b2) * g * g;
      param[j] += lr * (m[j] / c1) / (Math.sqrt(v[j] / c2) + eps);
    }
  }

  function greedy() {
    let prev = BOSI, out = '';
    for (let t = 0; t < L; t++) {
      const offP = t * V, offB = prev * V;
      let bestC = 0, bestZ = -Infinity;
      for (let c = 0; c < V; c++) {
        const z = Wp[offP + c] + Wb[offB + c];
        if (z > bestZ) { bestZ = z; bestC = c; }
      }
      out += vocab[bestC];
      prev = bestC;
    }
    return out;
  }

  const X = new Int16Array(batch * L);
  const prevArr = new Int16Array(batch);
  const R = new Float64Array(batch);
  const A = new Float64Array(batch);
  const history = [];
  let hitStep = null;

  for (let step = 0; step <= steps; step++) {
    // 1) 采样：按 (t, prev) 分组，每组只算一次 softmax
    prevArr.fill(BOSI);
    let groupsByT = null;
    if (uniform) {
      // 全 0 奖励下策略梯度为 0，均匀分布处熵梯度也为 0：策略始终均匀。
      // 直接均匀采样与 softmax 采样等价，但省掉 13×64×373 次 exp。
      for (let b = 0; b < batch; b++) {
        for (let t = 0; t < L; t++) X[b * L + t] = (rand() * V) | 0;
      }
    } else {
      groupsByT = new Array(L);
      for (let t = 0; t < L; t++) {
        const byPrev = new Map();
        for (let b = 0; b < batch; b++) {
          const p = prevArr[b];
          let arr = byPrev.get(p);
          if (!arr) { arr = []; byPrev.set(p, arr); }
          arr.push(b);
        }
        const groups = [];
        for (const [prev, idxs] of byPrev) {
          const pv = new Float32Array(V);
          const offP = t * V, offB = prev * V;
          let max = -Infinity;
          for (let c = 0; c < V; c++) {
            const z = Wp[offP + c] + Wb[offB + c];
            pv[c] = z;
            if (z > max) max = z;
          }
          let sum = 0;
          for (let c = 0; c < V; c++) { const e = Math.exp(pv[c] - max); pv[c] = e; sum += e; }
          const inv = 1 / sum, cdf = new Float32Array(V);
          let acc = 0;
          for (let c = 0; c < V; c++) { pv[c] *= inv; acc += pv[c]; cdf[c] = acc; }
          for (const b of idxs) {
            const u = rand();
            let lo = 0, hi = V - 1;
            while (lo < hi) {
              const mid = (lo + hi) >> 1;
              if (cdf[mid] < u) lo = mid + 1; else hi = mid;
            }
            X[b * L + t] = lo;
            prevArr[b] = lo;
          }
          groups.push({ prev, idxs, pv });
        }
        groupsByT[t] = groups;
      }
    }

    const texts = new Array(batch);
    for (let b = 0; b < batch; b++) texts[b] = decode(X, b * L, L, vocab);

    // 2) 奖励
    let meanR = 0, meanFlu = 0, meanMat = 0;
    for (let b = 0; b < batch; b++) {
      const s = texts[b];
      const flu = lm.fluency(s);
      const mat = matchCount(s, TARGET);
      meanFlu += flu; meanMat += mat;
      if (mode === 'fluency') R[b] = flu;
      else if (mode === 'target') R[b] = mat;
      else if (mode === 'sparse') R[b] = s === TARGET ? 1 : 0;
      else R[b] = flu + wTarget * mat;
      meanR += R[b];
      if (hitStep === null && s === TARGET) hitStep = step;
    }
    meanR /= batch; meanFlu /= batch; meanMat /= batch;
    if (!uniform) {
      let varR = 0;
      for (let b = 0; b < batch; b++) varR += (R[b] - meanR) ** 2;
      const stdR = Math.sqrt(varR / batch);
      for (let b = 0; b < batch; b++) A[b] = (R[b] - meanR) / (stdR + 1e-8);
    }

    // 3) 记录
    if (step % logEvery === 0) {
      let bestB = 0;
      for (let b = 1; b < batch; b++) if (R[b] > R[bestB]) bestB = b;
      const gs = greedy();
      history.push({
        step, reward: meanR, fluency: meanFlu, match: meanMat,
        best: texts[bestB], best_match: matchCount(texts[bestB], TARGET),
        greedy: gs, greedy_match: matchCount(gs, TARGET),
        greedy_ppl: lm.perplexity(gs), distinct2: distinct2(texts),
        samples: texts.slice(0, 6),
      });
    }
    if (step === steps) break;
    if (onProgress && step % 10 === 0) onProgress(step, steps);
    if (uniform) continue; // 策略保持均匀，不需要梯度

    // 4) 策略梯度：按 (t, prev) 分组求和，等价于逐样本梯度
    adamT++;
    const gWbMap = new Map();
    for (let t = 0; t < L; t++) {
      const gWp = new Float32Array(V);
      for (const { prev, idxs, pv } of groupsByT[t]) {
        let sumA = 0;
        const delta = new Float32Array(V);
        for (const b of idxs) {
          const a = A[b];
          sumA += a;
          delta[X[b * L + t]] += a;
        }
        let H = 0;
        for (let c = 0; c < V; c++) if (pv[c] > 0) H -= pv[c] * Math.log(pv[c]);
        const entScale = entropy * idxs.length;
        let row = gWbMap.get(prev);
        if (!row) { row = new Float32Array(V); gWbMap.set(prev, row); }
        for (let c = 0; c < V; c++) {
          const logp = Math.log(pv[c] + 1e-12);
          const g = -pv[c] * sumA + delta[c] + entScale * (-pv[c] * (logp + H));
          gWp[c] += g;
          row[c] += g;
        }
      }
      for (let c = 0; c < V; c++) gWp[c] /= batch;
      adam(Wp, t * V, gWp, mWp, vWp);
    }
    for (const [prev, row] of gWbMap) {
      for (let c = 0; c < V; c++) row[c] /= batch;
      adam(Wb, prev * V, row, mWb, vWb);
    }
  }

  const finalGreedy = greedy();
  return {
    name: `强化学习猴子（${mode}）`, mode, hit_step: hitStep,
    final_greedy: finalGreedy, final_match: matchCount(finalGreedy, TARGET),
    final_ppl: lm.perplexity(finalGreedy), history,
  };
}

// ---------------------------------------------------------------- Worker 入口
self.onmessage = (e) => {
  const { seed, corpus, target } = e.data;
  TARGET = (target || DEFAULT_TARGET).trim();
  const t0 = performance.now();
  const report = (pct, label) => self.postMessage({ type: 'progress', pct: Math.round(pct), label });
  try {
    const vocab = buildVocab(corpus, TARGET);
    const lm = new CharTrigramLM(corpus, vocab);
    const results = { target: TARGET, vocab_size: vocab.length, vocab: vocab.join(''), seed };

    report(2, '① 随机');
    results.exp1 = runExp1(vocab, mulberry32(seed));
    report(12, '① 随机');

    report(14, '② 黄鼠狼');
    results.exp2 = runExp2(vocab, mulberry32((seed ^ 0x9E3779B9) >>> 0));
    report(20, '② 黄鼠狼');

    const modes = ['sparse', 'fluency', 'target', 'mixed'];
    const labels = { sparse: '稀疏', fluency: '通顺', target: '目标', mixed: '混合' };
    results.exp3 = {};
    for (let i = 0; i < modes.length; i++) {
      const mode = modes[i];
      const base = 20 + i * 20;
      results.exp3[mode] = runExp3(vocab, lm, mode, (seed + 17 * (i + 1)) >>> 0,
        (step, total) => report(base + (step / total) * 20, `③ ${labels[mode]}`));
      report(base + 20, `③ ${labels[mode]}`);
    }

    results.elapsed_ms = Math.round(performance.now() - t0);
    self.postMessage({ type: 'done', results });
  } catch (err) {
    self.postMessage({ type: 'error', message: String((err && err.stack) || err) });
  }
};
