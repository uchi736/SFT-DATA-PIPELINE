"""
Phase3: RAFT 合成（LLM 非依存・決定的）。

各 (質問, ゴールデン, CoT回答) について:
  - 確率 oracle_prob でゴールデン + k ディストラクタ、(1-oracle_prob) でディストラクタのみ
    （--absent-mode refuse では後者を拒否例にする。assemble_one の説明を参照）
  - doc_order をシャッフル（ゴールデン位置をランダム化）
  - user メッセージ = 文脈整形 + 質問、assistant = CoT回答

--distractor-mode retrieved では、無作為抽出の代わりに検索の順位（raft/retrieval.py）を使う:
  - 文脈 = 検索の上位 (k+1) 件。正解の文書が入っていれば回答例、入っていなければゴールデン抜きの例
    （oracle_prob の抽選はしない。抜けるかどうかは検索が決める）
  - 正解の内容を持つ別の文書（本文が同じ・回答の引用を含む）は外し、次の順位の文書を繰り上げる
  - 並びは既定で入れ替える（--doc-order rank で検索の順位のまま）。合成した質問は本文に字面が近く、
    リランカーが正解の文書をほぼ必ず 1 位に置く。順位のまま並べると
    「1 番目の文書が正解」という位置の手掛かりが残るため

例ごとに rng=random.Random(stable_seed(seed, example_id)) を使うため、
入力順・並行・resume に依存せず、同 seed で assembled.jsonl がバイト一致する。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # scripts/
from raft.schemas import AnswerRecord, AssembledRecord, Chunk, make_example_id  # noqa: E402
from raft.corpus import stable_seed  # noqa: E402
from raft import formatting  # noqa: E402


ABSENT_MODES = ("answer", "refuse")
DISTRACTOR_MODES = ("random", "retrieved")
DOC_ORDERS = ("shuffle", "rank")


def assemble_one(ans: AnswerRecord, sampler, k: int, oracle_prob: float, seed: int,
                 query_label: str = "質問", absent_mode: str = "answer",
                 context_format: str = "document",
                 user_instruction: str | None = None) -> AssembledRecord:
    """1件の回答から RAFT 学習例を決定的に合成する。

    absent_mode:
      "answer"（既定・RAFT 論文どおり）: ゴールデンを抜いた例も、答えは CoT 回答のまま。
      "refuse": ゴールデンを抜いた例は拒否例にする。文書数を回答例とそろえるため、ゴールデンを
                doc_order に「差し替える位置の印」として残し refuse=True を立てる。実際の差し替えと
                拒否文の作文は拒否例フェーズ（raft/refusal.py）で行う。
    どちらのモードでも、ゴールデンを入れる例（oracle_included=True）の出力は同一。
    """
    example_id = make_example_id(ans.question_id)
    rng = random.Random(stable_seed(seed, example_id))

    golden = Chunk(id=ans.chunk_id, text=ans.golden_text)
    distractors = sampler(ans.chunk_id, k, salt="d")

    include_oracle = rng.random() < oracle_prob
    if not include_oracle and not distractors:
        # 文書0では検索学習にならないためゴールデンを強制投入
        include_oracle = True
    refuse = absent_mode == "refuse" and not include_oracle

    docs = ([golden] + distractors) if (include_oracle or refuse) else list(distractors)
    rng.shuffle(docs)

    user_content = formatting.format_context(
        docs, ans.question, query_label=query_label,
        style=context_format, instruction=user_instruction)
    assistant_content = ans.cot_text

    return AssembledRecord(
        example_id=example_id,
        chunk_id=ans.chunk_id,
        question=ans.question,
        cot_text=ans.cot_text,
        distractor_ids=[d.id for d in distractors],
        oracle_included=include_oracle,
        doc_order=[d.id for d in docs],
        user_content=user_content,
        assistant_content=assistant_content,
        refuse=refuse,
    )


def assemble_retrieved(ans: AnswerRecord, ranked: list[str], index: dict, sampler, k: int,
                       seed: int, query_label: str = "質問", absent_mode: str = "answer",
                       context_format: str = "document", user_instruction: str | None = None,
                       mix_random: float = 0.0, doc_order: str = "shuffle") -> AssembledRecord:
    """検索の順位から 1 件の学習例を決定的に合成する（--distractor-mode retrieved）。

    ranked: その質問の検索結果（文書 ID、最終順位の順）。正解の文書を含んでも含まなくてもよい。
    index : 文書 ID → Chunk（コーパス全体）。

    文脈は上位 (k+1) 件。正解の文書がその中にあれば回答例（oracle_included=True）。無ければ:
      absent_mode="answer": 答えは CoT 回答のまま（RAFT 論文どおり）
      absent_mode="refuse": refuse=True を立て、拒否文の作文は拒否例フェーズに任せる。
                            無作為抽出のときと違い、doc_order にゴールデンは入らない（差し替えは不要）
    mix_random: 正解以外の各枠を、この確率で無作為の文書に置き換える（検索の上位だけに偏らせない）。
    doc_order : "shuffle"=並びを入れ替える / "rank"=検索の順位のまま。
    """
    example_id = make_example_id(ans.question_id)
    rng = random.Random(stable_seed(seed, example_id))
    n_ctx = k + 1
    quotes = formatting.answer_quotes(ans.cot_text)

    def usable(cid: str) -> bool:
        c = index.get(cid)
        return c is not None and not formatting.carries_answer(c.text, ans.golden_text, quotes)

    # 検索の順位から、正解の内容を持つ別の文書を外す（外した分は次の順位が繰り上がる）
    kept: list[str] = []
    dropped: list[str] = []
    for cid in ranked:
        if cid == ans.chunk_id:
            kept.append(cid)
        elif cid in kept or cid in dropped:
            continue
        elif usable(cid):
            kept.append(cid)
        elif cid in index:
            dropped.append(cid)
    golden_rank = kept.index(ans.chunk_id) if ans.chunk_id in kept else -1
    top = kept[:n_ctx]

    # 無作為の文書（検索結果が足りないときの埋め合わせと、--mix-random の置き換えに使う）
    spare = [c.id for c in sampler(ans.chunk_id, n_ctx + 8, salt="mix")
             if c.id not in top and usable(c.id)]
    while len(top) < n_ctx and spare:
        top.append(spare.pop(0))

    include_oracle = ans.chunk_id in top
    if not include_oracle and not top:
        top, include_oracle = [ans.chunk_id], True     # 文書 0 件では学習例にならない

    n_mixed = 0
    if mix_random > 0.0:
        for i, cid in enumerate(top):
            if cid != ans.chunk_id and spare and rng.random() < mix_random:
                top[i] = spare.pop(0)
                n_mixed += 1

    if doc_order == "shuffle":
        rng.shuffle(top)
    refuse = absent_mode == "refuse" and not include_oracle

    docs = [Chunk(id=ans.chunk_id, text=ans.golden_text) if cid == ans.chunk_id else index[cid]
            for cid in top]
    user_content = formatting.format_context(
        docs, ans.question, query_label=query_label,
        style=context_format, instruction=user_instruction)

    return AssembledRecord(
        example_id=example_id,
        chunk_id=ans.chunk_id,
        question=ans.question,
        cot_text=ans.cot_text,
        distractor_ids=[cid for cid in top if cid != ans.chunk_id],
        oracle_included=include_oracle,
        doc_order=list(top),
        user_content=user_content,
        assistant_content=ans.cot_text,
        refuse=refuse,
        distractor_mode="retrieved",
        golden_rank=golden_rank,
        n_mixed=n_mixed,
        dropped_ids=dropped,
    )


def run_phase3(
    answers: list[AnswerRecord],
    sampler,
    k: int,
    oracle_prob: float,
    seed: int,
    query_label: str = "質問",
    absent_mode: str = "answer",
    context_format: str = "document",
    user_instruction: str | None = None,
    distractor_mode: str = "random",
    ranked_by_qid: dict | None = None,
    index: dict | None = None,
    mix_random: float = 0.0,
    doc_order: str = "shuffle",
) -> list[AssembledRecord]:
    """成功した回答 (ok=True かつ cot_text あり) を RAFT 例へ合成。

    distractor_mode="retrieved" では ranked_by_qid（question_id → 文書 ID の順位列）と index が要る。
    doc_order は retrieved のときだけ効く（random は従来どおり常に入れ替える）。
    """
    if absent_mode not in ABSENT_MODES:
        raise ValueError(f"未知の absent_mode: {absent_mode}")
    if distractor_mode not in DISTRACTOR_MODES:
        raise ValueError(f"未知の distractor_mode: {distractor_mode}")
    if doc_order not in DOC_ORDERS:
        raise ValueError(f"未知の doc_order: {doc_order}")
    if distractor_mode == "retrieved" and (ranked_by_qid is None or index is None):
        raise ValueError("distractor_mode=retrieved には ranked_by_qid と index が必要です")
    out: list[AssembledRecord] = []
    for ans in answers:
        if not ans.ok or not ans.cot_text.strip():
            continue
        if distractor_mode == "retrieved":
            ranked = ranked_by_qid.get(ans.question_id)
            if ranked is None:
                raise ValueError(f"検索結果がありません: {ans.question_id}")
            out.append(assemble_retrieved(
                ans, ranked, index, sampler, k, seed, query_label=query_label,
                absent_mode=absent_mode, context_format=context_format,
                user_instruction=user_instruction, mix_random=mix_random,
                doc_order=doc_order))
            continue
        out.append(assemble_one(
            ans, sampler, k, oracle_prob, seed, query_label=query_label,
            absent_mode=absent_mode, context_format=context_format,
            user_instruction=user_instruction))
    return out
