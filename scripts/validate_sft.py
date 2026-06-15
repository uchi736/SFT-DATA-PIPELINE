"""
出力 SFT JSONL をトレーナー契約に対して検証するユーティリティ。

    python scripts/validate_sft.py data/output/sft_dataset.jsonl [--max-len 2048] [--show 1]

チェック: messages 構造 / role が {system,user,assistant} / user→assistant 順 / 非空 /
assistant に ##begin_quote## 引用 / Swallow トークン長 <= max_len。
--show N で先頭 N 件の assistant 全文を表示。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # scripts/
from prompts import QUOTE_OPEN, QUOTE_CLOSE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--no-token", action="store_true", help="トークン長チェックを省略")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.path, encoding="utf-8") if l.strip()]
    counter = None
    if not args.no_token:
        from raft.tokenizer_utils import count_chat_tokens
        counter = count_chat_tokens

    n_quote = 0
    max_tok = 0
    for i, r in enumerate(recs):
        msgs = r.get("messages")
        assert isinstance(msgs, list) and msgs, f"行{i}: messages 不正"
        roles = [m["role"] for m in msgs]
        assert set(roles) <= {"system", "user", "assistant"}, f"行{i}: role={roles}"
        assert roles.index("user") < roles.index("assistant"), f"行{i}: 順序不正"
        assert all(m["content"].strip() for m in msgs), f"行{i}: 空 content"
        asst = next(m["content"] for m in msgs if m["role"] == "assistant")
        if QUOTE_OPEN in asst and QUOTE_CLOSE in asst:
            n_quote += 1
        if counter:
            t = counter(msgs)
            max_tok = max(max_tok, t)
            assert t <= args.max_len, f"行{i}: {t} > max_len {args.max_len}"

    print(f"OK: {len(recs)}件 / 引用あり {n_quote}件" + (f" / 最大トークン {max_tok}" if counter else ""))

    for i in range(min(args.show, len(recs))):
        asst = next(m["content"] for m in recs[i]["messages"] if m["role"] == "assistant")
        print(f"\n--- 例{i + 1} assistant 全文 ---\n{asst}")


if __name__ == "__main__":
    main()
