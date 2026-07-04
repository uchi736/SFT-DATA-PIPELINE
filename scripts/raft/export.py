"""
Phase4: エクスポート（長さゲート・重複排除・train/val 分割・書き出し）。

トレーナー契約に従い {"messages": [...]} の JSONL を出力する。長さは学習側 Swallow
トークナイザで apply_chat_template 後に計測（tokenizer_utils）。

oversize-policy:
  - drop  : max_len 超過例を破棄
  - reduce: ディストラクタを末尾から減らして収まる構成を採用（ゴールデンは最後まで残す）。
            min_distractors まで減らしても超過なら破棄。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))  # scripts/
from raft.schemas import AssembledRecord, Chunk, SFTRecord  # noqa: E402
from raft import formatting  # noqa: E402
from raft.corpus import stable_seed  # noqa: E402


def build_messages(user_content: str, assistant_content: str, system_prompt: Optional[str]) -> list[dict]:
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_content})
    msgs.append({"role": "assistant", "content": assistant_content})
    return msgs


def _reduce_to_fit(
    rec: AssembledRecord,
    index: dict[str, Chunk],
    counter: Callable[[list[dict]], int],
    max_len: int,
    min_distractors: int,
    system_prompt: Optional[str],
    query_label: str = "質問",
) -> tuple[list[dict], int, bool]:
    """ディストラクタを末尾から減らして収める。戻り値: (messages, token_count, reduced?)"""
    golden_in = rec.chunk_id if rec.oracle_included else None
    order = list(rec.doc_order)

    def num_distractors(o: list[str]) -> int:
        return sum(1 for i in o if i != golden_in)

    while True:
        docs = [index[i] for i in order if i in index]
        user = formatting.format_context(docs, rec.question, query_label=query_label)
        msgs = build_messages(user, rec.assistant_content, system_prompt)
        n = counter(msgs)
        reduced = len(order) != len(rec.doc_order)
        if n <= max_len or num_distractors(order) <= min_distractors:
            return msgs, n, reduced
        # 末尾側のディストラクタを1件除去
        removed = False
        for i in range(len(order) - 1, -1, -1):
            if order[i] != golden_in:
                del order[i]
                removed = True
                break
        if not removed:
            return msgs, n, reduced


def run_phase4(
    assembled: list[AssembledRecord],
    index: dict[str, Chunk],
    counter: Callable[[list[dict]], int],
    max_len: int = 2048,
    oversize_policy: str = "reduce",
    min_distractors: int = 1,
    system_prompt: Optional[str] = None,
    system_dropout: float = 0.0,
    seed: int = 42,
    query_label: str = "質問",
) -> tuple[list[SFTRecord], dict]:
    """長さゲート + 最終ペア重複排除して SFTRecord 列を返す。

    system_dropout>0 の場合、例ごとに決定的に一定割合を system 無しで出力する
    （推論時の system プロンプトのブレに対する頑健化）。
    """
    kept: list[SFTRecord] = []
    seen_pairs: set = set()
    stats = {
        "total": len(assembled),
        "kept": 0,
        "dropped_oversize": 0,
        "dropped_dup": 0,
        "reduced": 0,
        "no_system": 0,
    }

    for rec in assembled:
        # system ドロップアウト（例ごと決定的）
        sys_p = system_prompt
        if system_prompt and system_dropout > 0.0:
            rng = random.Random(stable_seed(seed, rec.example_id, "sysdrop"))
            if rng.random() < system_dropout:
                sys_p = None
        if sys_p is None and system_prompt:
            stats["no_system"] += 1

        msgs = build_messages(rec.user_content, rec.assistant_content, sys_p)
        n = counter(msgs)

        if n > max_len and oversize_policy == "reduce":
            msgs, n, reduced = _reduce_to_fit(
                rec, index, counter, max_len, min_distractors, sys_p, query_label
            )
            if reduced:
                stats["reduced"] += 1

        if n > max_len:
            stats["dropped_oversize"] += 1
            continue

        pair_key = (msgs[-2]["content"], msgs[-1]["content"])
        if pair_key in seen_pairs:
            stats["dropped_dup"] += 1
            continue
        seen_pairs.add(pair_key)
        kept.append(SFTRecord(messages=msgs))

    stats["kept"] = len(kept)
    return kept, stats


def _write_jsonl(records: list[SFTRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.model_dump(), ensure_ascii=False) + "\n")


def write_outputs(
    records: list[SFTRecord],
    output_path: str | Path,
    val_split: float = 0.0,
    seed: int = 42,
) -> dict:
    """sft_dataset.jsonl を書き出す。val_split>0 で _train/_val も作る（決定的）。"""
    output_path = Path(output_path)
    _write_jsonl(records, output_path)
    paths = {"all": str(output_path)}

    if val_split and 0.0 < val_split < 1.0 and len(records) > 1:
        idx = list(range(len(records)))
        random.Random(seed).shuffle(idx)
        n_val = max(1, int(len(records) * val_split))
        val_idx = set(idx[:n_val])
        train = [records[i] for i in idx[n_val:]]
        val = [records[i] for i in idx[:n_val]]
        train_path = output_path.with_name(output_path.stem + "_train" + output_path.suffix)
        val_path = output_path.with_name(output_path.stem + "_val" + output_path.suffix)
        _write_jsonl(train, train_path)
        _write_jsonl(val, val_path)
        paths["train"] = str(train_path)
        paths["val"] = str(val_path)

    return paths
