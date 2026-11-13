"""一键运行全部实验：导出 web/data/results.json 供 3D 可视化回放，并画报告用图。

用法：py run_all.py            （完整，约 1~2 分钟）
      py run_all.py --quick    （快速版）
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from monkey import exp1_random, exp2_weasel, exp3_rl
from monkey.common import ROOT, TARGET, CharTrigramLM, build_vocab, load_corpus


def thin(history, max_points=400):
    """回放数据抽稀，但保留首尾。"""
    if len(history) <= max_points:
        return history
    idx = np.unique(np.linspace(0, len(history) - 1, max_points).astype(int))
    return [history[i] for i in idx]


def plots(res, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False
    outdir.mkdir(parents=True, exist_ok=True)

    # 图1：随机猴子命中字数分布 vs 二项分布理论值
    e1 = res["exp1"]
    h = np.array(e1["match_histogram"], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ks = np.arange(len(h))
    ax.bar(ks, h / h.sum(), label="实验")
    from math import comb
    p = 1 / e1["theory"]["vocab_size"]
    ax.plot(ks, [comb(13, k) * p**k * (1 - p)**(13 - k) for k in ks], "o-", color="C3", label="二项分布理论")
    ax.set_yscale("log"); ax.set_xlabel("一次尝试中位置正确的字数"); ax.set_ylabel("频率")
    ax.set_title(f"实验① 随机猴子：{e1['attempts']:,} 次尝试，最多对 {e1['best_match']} 个字")
    ax.legend(); fig.tight_layout(); fig.savefig(outdir / "exp1_random.png", dpi=150); plt.close(fig)

    # 图2：黄鼠狼适应度曲线 + 参数扫描热力图
    e2 = res["exp2"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    axs[0].plot([x["gen"] for x in e2["history"]], [x["fitness"] for x in e2["history"]])
    axs[0].set_xlabel("代数"); axs[0].set_ylabel("正确字数"); axs[0].set_title(f"实验② 累积选择：{e2['generations']} 代命中")
    sw = e2["sweep"]
    pops = sorted({x["pop"] for x in sw}); mus = sorted({x["mu"] for x in sw})
    M = np.array([[next(x["mean_gen"] for x in sw if x["pop"] == p and x["mu"] == m) for m in mus] for p in pops])
    im = axs[1].imshow(np.log10(M), cmap="viridis_r", aspect="auto")
    axs[1].set_xticks(range(len(mus)), mus); axs[1].set_yticks(range(len(pops)), pops)
    axs[1].set_xlabel("变异率 μ"); axs[1].set_ylabel("种群大小")
    for i in range(len(pops)):
        for j in range(len(mus)):
            axs[1].text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", color="w", fontsize=8)
    axs[1].set_title("平均命中代数（色标 log10）"); fig.colorbar(im, ax=axs[1])
    fig.tight_layout(); fig.savefig(outdir / "exp2_weasel.png", dpi=150); plt.close(fig)

    # 图3：强化学习，不同奖励设计对比
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    for mode, r in res["exp3"].items():
        st = [x["step"] for x in r["history"]]
        axs[0].plot(st, [x["match"] for x in r["history"]], label=mode)
        axs[1].plot(st, [x["fluency"] for x in r["history"]], label=mode)
        axs[2].plot(st, [x["distinct2"] for x in r["history"]], label=mode)
    for ax, t in zip(axs, ["平均正确字数（像不像目标句）", "平均每字对数概率（通顺度）", "distinct-2（多样性）"]):
        ax.set_title(t); ax.set_xlabel("训练步"); ax.legend()
    fig.tight_layout(); fig.savefig(outdir / "exp3_rl.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    corpus = load_corpus()
    vocab = build_vocab(corpus)
    lm = CharTrigramLM(corpus, vocab)
    print(f"键盘字数 V={len(vocab)}，目标句「{TARGET}」，目标句困惑度={lm.perplexity(TARGET):.1f}")

    res = {"target": TARGET, "vocab_size": len(vocab), "vocab": "".join(vocab)}

    print("实验① 纯随机猴子 ...")
    e1 = exp1_random.run(vocab, attempts=400_000 if a.quick else 4_000_000)
    print(f"  {e1['attempts']:,} 次，最好只对了 {e1['best_match']} 个字：{e1['best_string']}")
    print(f"  理论：搜索空间 10^{e1['theory']['log10_search_space']:.1f}，每秒 10^9 次也要 "
          f"10^{e1['theory']['log10_universe_ages']:.1f} 个宇宙年龄")
    e1["history"] = thin(e1["history"], 200)
    res["exp1"] = e1

    print("实验② 黄鼠狼进化 ...")
    e2 = exp2_weasel.run(vocab)
    print(f"  第 {e2['generations']} 代命中（共评估 {e2['evaluations']:,} 个串）")
    e2["history"] = thin(e2["history"], 400)
    e2["sweep"] = exp2_weasel.sweep(vocab, repeats=2 if a.quick else 5)
    res["exp2"] = e2

    print("实验③ 强化学习猴子 ...")
    steps = 300 if a.quick else 600
    res["exp3"] = {}
    for mode in ["sparse", "fluency", "target", "mixed"]:
        r = exp3_rl.run(vocab, lm, mode=mode, steps=steps, log_every=2)
        r["history"] = thin(r["history"], 300)
        res["exp3"][mode] = r
        print(f"  [{mode:7s}] 首次命中步={r['hit_step']}，最终贪心输出「{r['final_greedy']}」"
              f"对 {r['final_match']}/13 字，困惑度 {r['final_ppl']:.1f}")

    seeds = range(3 if a.quick else 8)
    res["exp3_seeds"] = {
        m: [exp3_rl.run(vocab, lm, mode=m, steps=steps, log_every=10_000, seed=s, stop_on_hit=True)["hit_step"]
            for s in seeds] for m in ["target", "mixed"]}
    for m, v in res["exp3_seeds"].items():
        hits = [x for x in v if x is not None]
        print(f"  多种子 [{m}] 命中步：{v}，均值 {np.mean(hits) if hits else float('nan'):.0f}")

    out = ROOT / "web" / "data"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    # 网页里的“实时训练”Worker 也要读这份小语料
    (out / "corpus.txt").write_text((ROOT / "data" / "corpus.txt").read_text(encoding="utf-8"), encoding="utf-8")
    plots(res, ROOT / "outputs" / "figures")
    print(f"完成，用时 {time.time() - t0:.0f}s → web/data/results.json, web/data/corpus.txt, outputs/figures/*.png")


if __name__ == "__main__":
    main()
