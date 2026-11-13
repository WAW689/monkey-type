"""诊断脚本：复现 mixed 奖励在 600 步内未命中的 5 个种子。

运行：py tools/diag_mixed.py
输出：outputs/mixed_failures.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from monkey import exp3_rl  # noqa: E402
from monkey.common import TARGET, CharTrigramLM, build_vocab, load_corpus  # noqa: E402

corpus = load_corpus()
vocab = build_vocab(corpus)
lm = CharTrigramLM(corpus, vocab)

out = []
for seed in (0, 1, 5, 6, 7):
    r = exp3_rl.run(vocab, lm, mode="mixed", steps=600, log_every=10_000,
                    seed=seed, stop_on_hit=True)
    row = {
        "seed": seed,
        "hit": r["hit_step"],
        "greedy": r["final_greedy"],
        "match": r["final_match"],
        "ppl": round(r["final_ppl"], 2),
        "reward_target": round(lm.fluency(TARGET) + 0.6 * len(TARGET), 3),
        "reward_greedy": round(lm.fluency(r["final_greedy"]) + 0.6 * r["final_match"], 3),
    }
    out.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)

path = ROOT / "outputs" / "mixed_failures.json"
path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"→ {path}")
