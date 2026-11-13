"""实验②：黄鼠狼程序（累积选择 / 最简遗传算法）。

每一代：以当前最优串为父本，复制 pop 份，每个字以 mu 的概率变异为随机字；
保留适应度（逐位匹配数）最高者（含父本，即精英保留）。
"""
from __future__ import annotations

import numpy as np

from .common import TARGET, rng


def run(vocab: list[str], pop: int = 100, mu: float = 0.04, max_gen: int = 20_000,
        seed: int = 2, record: bool = True) -> dict:
    g = rng(seed)
    V, L = len(vocab), len(TARGET)
    tgt = np.array([vocab.index(c) for c in TARGET])
    parent = g.integers(0, V, size=L)
    history = []
    gen = 0
    for gen in range(max_gen + 1):
        fit = int((parent == tgt).sum())
        if record:
            history.append({"gen": gen, "best": "".join(vocab[j] for j in parent), "fitness": fit})
        if fit == L:
            break
        kids = np.repeat(parent[None, :], pop, axis=0)
        mask = g.random((pop, L)) < mu
        kids[mask] = g.integers(0, V, size=int(mask.sum()))
        kids = np.vstack([parent[None, :], kids])
        parent = kids[int((kids == tgt).sum(axis=1).argmax())]
    hit = int((parent == tgt).sum()) == L
    return {"name": "黄鼠狼进化", "pop": pop, "mu": mu, "hit": hit,
            "generations": gen, "evaluations": gen * pop, "history": history}


def sweep(vocab: list[str], pops=(10, 50, 100, 500), mus=(0.01, 0.02, 0.04, 0.08, 0.16, 0.3),
          repeats: int = 5, max_gen: int = 20_000) -> list[dict]:
    """参数扫描：种群大小 × 变异率 → 平均命中代数（报告里画热力图用）。"""
    out = []
    for p in pops:
        for m in mus:
            gens = [run(vocab, p, m, max_gen, seed=100 + r, record=False) for r in range(repeats)]
            out.append({"pop": p, "mu": m,
                        "mean_gen": float(np.mean([x["generations"] for x in gens])),
                        "hit_rate": float(np.mean([x["hit"] for x in gens]))})
    return out
