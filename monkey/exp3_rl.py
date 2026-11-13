"""实验③：强化学习猴子（REINFORCE 策略梯度）。

策略：对数线性 softmax，下一个字的打分 = W_pos[位置] + W_prev[上一个字]
        π(c | t, prev) = softmax(W_pos[t] + W_prev[prev])
奖励（三种模式，用于对比奖励设计）：
  fluency : 只奖励"通顺"——三元语言模型给出的每字平均对数概率
  target  : 只奖励"像目标句"——逐位匹配数
  mixed   : 两者加权
  sparse  : 只有完全打对才给 1 分（稀疏奖励，演示"学不动"）
优势 A = (R - 批均值) / 批标准差：高于平均的句子被"激励"（概率上调），
低于平均的被"惩罚"（概率下调）。
"""
from __future__ import annotations

import numpy as np

from .common import TARGET, CharTrigramLM, distinct_n, match_count, rng


class Adam:
    def __init__(self, shapes, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]

    def step(self, params, grads):  # 梯度上升
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m[:] = self.b1 * m + (1 - self.b1) * g
            v[:] = self.b2 * v + (1 - self.b2) * g * g
            mh, vh = m / (1 - self.b1 ** self.t), v / (1 - self.b2 ** self.t)
            p += self.lr * mh / (np.sqrt(vh) + self.eps)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def run(vocab: list[str], lm: CharTrigramLM, mode: str = "mixed", steps: int = 3000,
        batch: int = 64, lr: float = 0.05, w_target: float = 0.6, entropy: float = 0.01,
        log_every: int = 10, seed: int = 3, stop_on_hit: bool = False) -> dict:
    g = rng(seed)
    V, L = len(vocab), len(TARGET)
    BOS = V
    Wp = np.zeros((L, V))
    Wb = np.zeros((V + 1, V))
    opt = Adam([Wp.shape, Wb.shape], lr=lr)
    history, hit_step = [], None

    def decode(idx):
        return "".join(vocab[j] for j in idx)

    def greedy():
        prev, out = BOS, []
        for t in range(L):
            c = int((Wp[t] + Wb[prev]).argmax())
            out.append(c)
            prev = c
        return decode(out)

    for step in range(steps + 1):
        # 1) 采样一批句子（猴子打字）
        X = np.zeros((batch, L), dtype=int)
        P = np.zeros((batch, L, V))
        prev = np.full(batch, BOS)
        for t in range(L):
            p = softmax(Wp[t][None, :] + Wb[prev])
            c = (p.cumsum(axis=1) > g.random((batch, 1))).argmax(axis=1)
            X[:, t], P[:, t] = c, p
            prev = c
        texts = [decode(row) for row in X]

        # 2) 打分
        flu = np.array([lm.fluency(s) for s in texts])
        mat = np.array([match_count(s) for s in texts], dtype=float)
        if mode == "fluency":
            R = flu
        elif mode == "target":
            R = mat
        elif mode == "sparse":  # 只有一字不差才给奖励：几乎永远是 0，学不到东西
            R = np.array([float(x == TARGET) for x in texts])
        else:
            R = flu + w_target * mat
        A = (R - R.mean()) / (R.std() + 1e-8)  # >0 激励，<0 惩罚

        if hit_step is None and any(s == TARGET for s in texts):
            hit_step = step

        if step % log_every == 0:
            k = int(R.argmax())
            gs = greedy()
            history.append({
                "step": step, "reward": float(R.mean()), "fluency": float(flu.mean()),
                "match": float(mat.mean()), "best": texts[k], "best_match": int(mat[k]),
                "greedy": gs, "greedy_match": match_count(gs),
                "greedy_ppl": lm.perplexity(gs), "distinct2": distinct_n(texts, 2),
                "samples": texts[:6],
            })
        if step == steps or (stop_on_hit and hit_step is not None):
            break

        # 3) 策略梯度：∇log π(a) = onehot(a) − π
        G = -P * A[:, None, None]
        np.add.at(G, (np.arange(batch)[:, None], np.arange(L)[None, :], X), A[:, None])
        # 熵正则，防止过早坍缩：∇H = −π(log π + H)
        logP = np.log(P + 1e-12)
        H = -(P * logP).sum(-1, keepdims=True)
        G += entropy * (-P * (logP + H))
        G /= batch
        gWp = G.sum(axis=0)
        prevs = np.concatenate([np.full((batch, 1), BOS), X[:, :-1]], axis=1)
        gWb = np.zeros_like(Wb)
        np.add.at(gWb, prevs.reshape(-1), G.reshape(-1, V))
        opt.step([Wp, Wb], [gWp, gWb])

    final = greedy()
    return {"name": f"强化学习猴子（{mode}）", "mode": mode, "hit_step": hit_step,
            "final_greedy": final, "final_match": match_count(final),
            "final_ppl": lm.perplexity(final), "history": history}
