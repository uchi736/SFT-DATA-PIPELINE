"""
生成コスト/スループット見積（任意・ネット不要）。

コーパスと生成パラメータから LLM 呼び出し回数と概算所要時間を見積もる。
vLLM はローカル従量課金が無いため pricing ではなく所要時間の目安を主に出す。
Azure 等へ切替時のために、--price-in/--price-out を与えると概算コストも表示する。

    python scripts/raft/estimate.py data/input/corpus.jsonl \
        --questions-per-chunk 3 --concurrency 8 --sec-per-call 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # scripts/
from raft import corpus as corpus_mod  # noqa: E402


def estimate(
    n_golden: int,
    questions_per_chunk: int,
    concurrency: int,
    sec_per_call: float,
) -> dict:
    phase1_calls = n_golden
    phase2_calls = n_golden * questions_per_chunk
    total_calls = phase1_calls + phase2_calls
    # 並行実行を考慮した概算 wall-clock（秒）
    wall = (total_calls / max(1, concurrency)) * sec_per_call
    return {
        "phase1_calls": phase1_calls,
        "phase2_calls": phase2_calls,
        "total_calls": total_calls,
        "est_wall_clock_sec": round(wall, 1),
        "est_wall_clock_min": round(wall / 60, 1),
    }


def main():
    p = argparse.ArgumentParser(
        description="RAFT 生成コスト/スループット見積",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input")
    p.add_argument("--questions-per-chunk", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--sec-per-call", type=float, default=4.0, help="1 LLM 呼び出しの平均秒数")
    p.add_argument("--min-chunk-chars", type=int, default=corpus_mod.MIN_CHUNK_CHARS)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    chunks = corpus_mod.load_corpus(args.input, limit=args.limit)
    goldens = corpus_mod.golden_candidates(chunks, min_chars=args.min_chunk_chars)
    est = estimate(len(goldens), args.questions_per_chunk, args.concurrency, args.sec_per_call)
    print(f"コーパス: 全{len(chunks)}件 / ゴールデン候補{len(goldens)}件")
    for k, v in est.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
