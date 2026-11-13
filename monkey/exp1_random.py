"""实验①：纯随机猴子。每次从键盘上均匀随机敲 len(TARGET) 个字。

结论应当是：理论上能打出，实际上永远打不出。
"""
from __future__ import annotations

import math

import numpy as np

from .common import TARGET, rng


def theory(vocab_size: int, length: int = len(TARGET), speed: float = 1e9) -> dict:
    log10_space = length * math.log10(vocab_size)
    age_universe_s = 4.35e17
    log10_seconds = log10_space - math.log10(speed)  # 期望尝试次数≈搜索空间大小
    return {
        "vocab_size": vocab_size,
        "length": length,
        "log10_search_space": log10_space,
        "attempts_per_second": speed,
        "log10_expected_seconds": log10_seconds,
        "log10_universe_ages": log10_seconds - math.log10(age_universe_s),
    }


def run(vocab: list[str], attempts: int = 2_000_000, batch: int = 200_000,
        log_every: int = 20_000, seed: int = 1) -> dict:
    g = rng(seed)
    V, L = len(vocab), len(TARGET)
    tgt = np.array([vocab.index(c) for c in TARGET])
    best, best_str, history, samples = -1, "", [], []
    hist_counts = np.zeros(L + 1, dtype=np.int64)  # 命中 k 个字的次数分布
    done = 0
    while done < attempts:
        x = g.integers(0, V, size=(batch, L))
        m = (x == tgt).sum(axis=1)
        hist_counts += np.bincount(m, minlength=L + 1)
        i = int(m.argmax())
        if m[i] > best:
            best, best_str = int(m[i]), "".join(vocab[j] for j in x[i])
        for k in range(0, batch, log_every):
            history.append({"attempt": done + k + log_every, "best": best,
                            "sample": "".join(vocab[j] for j in x[k])})
        if len(samples) < 300:
            samples += ["".join(vocab[j] for j in row) for row in x[:300 - len(samples)]]
        done += batch
    return {
        "name": "纯随机猴子",
        "attempts": done,
        "best_match": best,
        "best_string": best_str,
        "match_histogram": hist_counts.tolist(),
        "history": history,
        "samples": samples,
        "theory": theory(V),
    }
