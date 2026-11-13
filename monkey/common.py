"""公共部分：目标句、字符表、通顺度语言模型、相似度。"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TARGET = "生存还是毁灭，这是一个问题"
BOS, EOS = "^", "$"


def load_corpus(path: Path = ROOT / "data" / "corpus.txt") -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def build_vocab(corpus: list[str], target: str = TARGET) -> list[str]:
    """猴子的键盘 = 语料中出现过的字 ∪ 目标句的字（保证目标可达）。"""
    chars = set("".join(corpus)) | set(target)
    return sorted(chars)


class CharTrigramLM:
    """字符级三元语言模型，插值平滑（trigram/bigram/unigram/均匀分布）。

    fluency(s) 返回每字平均对数概率，越大越"通顺"。
    """

    def __init__(self, corpus: list[str], vocab: list[str], lambdas=(0.5, 0.3, 0.15, 0.05)):
        self.vocab = vocab
        self.V = len(vocab) + 1  # +EOS
        self.l3, self.l2, self.l1, self.l0 = lambdas
        self.uni, self.bi, self.tri = Counter(), Counter(), Counter()
        self.ctx1, self.ctx2 = Counter(), Counter()
        for s in corpus:
            seq = [BOS, BOS] + list(s) + [EOS]
            for i in range(2, len(seq)):
                a, b, c = seq[i - 2], seq[i - 1], seq[i]
                self.uni[c] += 1
                self.bi[(b, c)] += 1
                self.ctx1[b] += 1
                self.tri[(a, b, c)] += 1
                self.ctx2[(a, b)] += 1
        self.N = sum(self.uni.values())

    def prob(self, a: str, b: str, c: str) -> float:
        p3 = self.tri[(a, b, c)] / self.ctx2[(a, b)] if self.ctx2[(a, b)] else 0.0
        p2 = self.bi[(b, c)] / self.ctx1[b] if self.ctx1[b] else 0.0
        p1 = self.uni[c] / self.N
        return self.l3 * p3 + self.l2 * p2 + self.l1 * p1 + self.l0 / self.V

    def logprob(self, s: str, eos: bool = True) -> float:
        seq = [BOS, BOS] + list(s) + ([EOS] if eos else [])
        return sum(math.log(self.prob(seq[i - 2], seq[i - 1], seq[i])) for i in range(2, len(seq)))

    def fluency(self, s: str) -> float:
        return self.logprob(s) / (len(s) + 1)

    def perplexity(self, s: str) -> float:
        return math.exp(-self.fluency(s))


def match_count(s: str, target: str = TARGET) -> int:
    """逐位相同的字数（黄鼠狼程序的适应度）。"""
    return sum(a == b for a, b in zip(s, target))


def distinct_n(samples: list[str], n: int = 2) -> float:
    """输出多样性：不同 n-gram 数 / n-gram 总数。越低说明越"模式坍缩"。"""
    grams = [s[i:i + n] for s in samples for i in range(len(s) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)
