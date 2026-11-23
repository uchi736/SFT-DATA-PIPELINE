"""
Phase4: エクスポート（長さゲート・重複排除・train/val 分割・書き出し）。

トレーナー契約に従い {"messages": [...]} の JSONL を出力する。長さは学習側 Swallow
トークナイザで apply_chat_template 後に計測（tokenizer_utils）。

oversize-policy:
  - drop  : max_len 超過例を破棄
  - reduce: ディストラクタを末尾から減らして収まる構成を採用（ゴールデンは最後まで残す）。
            min_distractors まで減らしても超過なら破棄。

拒否例（--absent-mode refuse / --twin-ratio）を混ぜる場合:
  - 回答例の最終的な文書の並びは fit_record で決まる。拒否例はその並びを受け継いで作るため、
    書き出し時は長さの確認だけを行い、文書は減らさない（超過は破棄）。
  - train/val は質問単位で分ける（同じ質問の回答例と双子が train と val に分かれないように）。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))  # scripts/
from raft.schemas import AssembledRecord, Chunk, RefusalRecord, SFTRecord  # noqa: E402
from raft import formatting  # noqa: E402
from raft.corpus import stable_seed  # noqa: E402


def build_messages(user_content: str, assistant_content: str, system_prompt: Optional[str]) -> list[dict]:
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_content})
    msgs.append({"role": "assistant", "content": assistant_content})
    return msgs


def system_for(example_id: str, system_prompt: Optional[str], system_dropout: float, seed: int) -> Optional[str]:
    """例ごとに決定的な system ドロップアウトを適用した system を返す。"""
    if system_prompt and system_dropout > 0.0:
        rng = random.Random(stable_seed(seed, example_id, "sysdrop"))
        if rng.random() < system_dropout:
            return None
    return system_prompt


def _reduce_to_fit(
    rec: AssembledRecord,
    index: dict[str, Chunk],
    counter: Callable[[list[dict]], int],
    max_len: int,
    min_distractors: int,
    system_prompt: Optional[str],
    query_label: str = "質問",
    context_format: str = "document",
    user_instruction: Optional[str] = None,
) -> tuple[list[dict], int, bool, list[str]]:
    """ディストラクタを末尾から減らして収める。戻り値: (messages, token_count, reduced?, 文書 ID 列)

    ゴールデン（refuse=True の例では差し替える位置の印）は最後まで残す。
    """
    golden_in = rec.chunk_id if (rec.oracle_included or rec.refuse) else None
    order = list(rec.doc_order)

    def num_distractors(o: list[str]) -> int:
        return sum(1 for i in o if i != golden_in)

    while True:
        docs = [index[i] for i in order if i in index]
        user = formatting.format_context(docs, rec.question, query_label=query_label,
                                         style=context_format, instruction=user_instruction)
        msgs = build_messages(user, rec.assistant_content, system_prompt)
        n = counter(msgs)
        reduced = len(order) != len(rec.doc_order)
        if n <= max_len or num_distractors(order) <= min_distractors:
            return msgs, n, reduced, order
        # 末尾側のディストラクタを1件除去
        removed = False
        for i in range(len(order) - 1, -1, -1):
            if order[i] != golden_in:
                del order[i]
                removed = True
                break
        if not removed:
            return msgs, n, reduced, order


def fit_record(
    rec: AssembledRecord,
    index: dict[str, Chunk],
    counter: Callable[[list[dict]], int],
    max_len: int = 2048,
    oversize_policy: str = "reduce",
    min_distractors: int = 1,
    system_prompt: Optional[str] = None,
    system_dropout: float = 0.0,
    seed: int = 42,
    query_label: str = "質問",
    context_format: str = "document",
    user_instruction: Optional[str] = None,
) -> tuple[list[dict], int, bool, list[str], bool]:
    """1 例を長さ上限に収めた結果を返す: (messages, token_count, reduced?, 文書 ID 列, system 無し?)

    run_phase4 と拒否例の計画（refusal.plan_tasks）の両方がこれを使うので、
    回答例の最終的な並びと、それを受け継ぐ拒否例の並びが食い違わない。
    """
    sys_p = system_for(rec.example_id, system_prompt, system_dropout, seed)
    msgs = build_messages(rec.user_content, rec.assistant_content, sys_p)
    n = counter(msgs)
    reduced = False
    order = list(rec.doc_order)
    if n > max_len and oversize_policy == "reduce":
        msgs, n, reduced, order = _reduce_to_fit(
            rec, index, counter, max_len, min_distractors, sys_p, query_label,
            context_format, user_instruction,
        )
    return msgs, n, reduced, order, bool(system_prompt) and sys_p is None


def _new_stats(total: int) -> dict:
    return {"total": total, "kept": 0, "dropped_oversize": 0, "dropped_dup": 0,
            "reduced": 0, "no_system": 0}


def _export_assembled(assembled, index, counter, max_len, oversize_policy, min_distractors,
                      system_prompt, system_dropout, seed, query_label,
                      context_format, user_instruction, stats, seen_pairs):
    """回答例（と既定モードの oracle なし例）を書き出し用に整える。戻り値: [(SFTRecord, meta)]"""
    out = []
    for rec in assembled:
        if rec.refuse:
            # 差し替え前の並び（ゴールデン入り）なので書き出さない。拒否例は export_refusals が出す
            continue
        msgs, n, reduced, order, no_sys = fit_record(
            rec, index, counter, max_len=max_len, oversize_policy=oversize_policy,
            min_distractors=min_distractors, system_prompt=system_prompt,
            system_dropout=system_dropout, seed=seed, query_label=query_label,
            context_format=context_format, user_instruction=user_instruction,
        )
        if no_sys:
            stats["no_system"] += 1
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
        meta = {"example_id": rec.example_id, "group": rec.example_id,
                "kind": "answer" if rec.oracle_included else "absent_answer",
                "n_docs": len(order)}
        out.append((SFTRecord(messages=msgs), meta))
    return out


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
    context_format: str = "document",
    user_instruction: Optional[str] = None,
) -> tuple[list[SFTRecord], dict]:
    """長さゲート + 最終ペア重複排除して SFTRecord 列を返す。

    system_dropout>0 の場合、例ごとに決定的に一定割合を system 無しで出力する
    （推論時の system プロンプトのブレに対する頑健化）。
    """
    records, _, stats = run_phase4_mixed(
        assembled, [], index, counter, max_len=max_len, oversize_policy=oversize_policy,
        min_distractors=min_distractors, system_prompt=system_prompt,
        system_dropout=system_dropout, seed=seed, query_label=query_label,
        context_format=context_format, user_instruction=user_instruction,
    )
    return records, stats


def run_phase4_mixed(
    assembled: list[AssembledRecord],
    refusals: list[RefusalRecord],
    index: dict[str, Chunk],
    counter: Callable[[list[dict]], int],
    max_len: int = 2048,
    oversize_policy: str = "reduce",
    min_distractors: int = 1,
    system_prompt: Optional[str] = None,
    system_dropout: float = 0.0,
    seed: int = 42,
    query_label: str = "質問",
    context_format: str = "document",
    user_instruction: Optional[str] = None,
    shuffle: bool = False,
) -> tuple[list[SFTRecord], list[dict], dict]:
    """回答例と拒否例をまとめて書き出し用に整える。戻り値: (records, metas, stats)

    metas は records と同じ並びで、例ごとに {example_id, group(質問), kind, n_docs} を持つ。
    kind: answer / absent_answer（既定モードの oracle なし例）/ absent / twin
    refusals は status=ok かつ採用済みのものだけを渡す。
    """
    stats = _new_stats(len(assembled))
    seen_pairs: set = set()
    rows = _export_assembled(
        assembled, index, counter, max_len, oversize_policy, min_distractors,
        system_prompt, system_dropout, seed, query_label, context_format, user_instruction,
        stats, seen_pairs,
    )
    if refusals:
        stats.update({"refusals": len(refusals), "dropped_refusal_oversize": 0})
    for ref in refusals:
        # system の有無は質問単位でそろえる（双子は回答例と同じ判定になる）
        sys_p = system_for(ref.question_id, system_prompt, system_dropout, seed)
        docs = [index[i] for i in ref.doc_order]
        user = formatting.format_context(docs, ref.question, query_label=query_label,
                                         style=context_format, instruction=user_instruction)
        msgs = build_messages(user, ref.refusal, sys_p)
        if counter(msgs) > max_len:
            # 文書を減らすと作文時に見せた資料と食い違うので、減らさず捨てる
            stats["dropped_refusal_oversize"] += 1
            continue
        pair_key = (msgs[-2]["content"], msgs[-1]["content"])
        if pair_key in seen_pairs:
            stats["dropped_dup"] += 1
            continue
        seen_pairs.add(pair_key)
        if system_prompt and sys_p is None:
            stats["no_system"] += 1
        meta = {"example_id": ref.example_id, "group": ref.question_id,
                "kind": ref.kind, "n_docs": len(docs)}
        rows.append((SFTRecord(messages=msgs), meta))

    if shuffle:
        random.Random(stable_seed(seed, "export_shuffle")).shuffle(rows)
    stats["kept"] = len(rows)
    return [r for r, _ in rows], [m for _, m in rows], stats


def _write_jsonl(records: list[SFTRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.model_dump(), ensure_ascii=False) + "\n")


def split_indices(n: int, val_split: float, seed: int, groups: Optional[list[str]] = None):
    """(train の添字列, val の添字列) を決定的に返す。分けない場合は (全件, [])。

    groups を渡すと、同じ group の例は必ず同じ側に入る（質問単位の分割）。
    """
    if not (val_split and 0.0 < val_split < 1.0 and n > 1):
        return list(range(n)), []
    if groups is None:
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        n_val = max(1, int(n * val_split))
        return idx[n_val:], idx[:n_val]
    uniq = sorted(set(groups))
    random.Random(seed).shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_split))
    val_groups = set(uniq[:n_val])
    train = [i for i in range(n) if groups[i] not in val_groups]
    val = [i for i in range(n) if groups[i] in val_groups]
    return train, val


def write_outputs(
    records: list[SFTRecord],
    output_path: str | Path,
    val_split: float = 0.0,
    seed: int = 42,
    groups: Optional[list[str]] = None,
) -> dict:
    """sft_dataset.jsonl を書き出す。val_split>0 で _train/_val も作る（決定的）。

    groups（records と同じ並びの質問 ID 等）を渡すと、質問単位で train/val を分ける。
    """
    output_path = Path(output_path)
    _write_jsonl(records, output_path)
    paths = {"all": str(output_path)}

    train_idx, val_idx = split_indices(len(records), val_split, seed, groups)
    if val_idx:
        train_path = output_path.with_name(output_path.stem + "_train" + output_path.suffix)
        val_path = output_path.with_name(output_path.stem + "_val" + output_path.suffix)
        _write_jsonl([records[i] for i in train_idx], train_path)
        _write_jsonl([records[i] for i in val_idx], val_path)
        paths["train"] = str(train_path)
        paths["val"] = str(val_path)

    return paths
