"""
Phase3: RAFT 合成（LLM 非依存・決定的）。

各 (質問, ゴールデン, CoT回答) について:
  - 確率 oracle_prob でゴールデン + k ディストラクタ、(1-oracle_prob) でディストラクタのみ
  - doc_order をシャッフル（ゴールデン位置をランダム化）
  - user メッセージ = 文脈整形 + 質問、assistant = CoT回答

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


def assemble_one(ans: AnswerRecord, sampler, k: int, oracle_prob: float, seed: int) -> AssembledRecord:
    """1件の回答から RAFT 学習例を決定的に合成する。"""
    example_id = make_example_id(ans.question_id)
    rng = random.Random(stable_seed(seed, example_id))

    golden = Chunk(id=ans.chunk_id, text=ans.golden_text)
    distractors = sampler(ans.chunk_id, k, salt="d")

    include_oracle = rng.random() < oracle_prob
    if not include_oracle and not distractors:
        # 文書0では検索学習にならないためゴールデンを強制投入
        include_oracle = True

    docs = ([golden] + distractors) if include_oracle else list(distractors)
    rng.shuffle(docs)

    user_content = formatting.format_context(docs, ans.question)
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
    )


def run_phase3(
    answers: list[AnswerRecord],
    sampler,
    k: int,
    oracle_prob: float,
    seed: int,
) -> list[AssembledRecord]:
    """成功した回答 (ok=True かつ cot_text あり) を RAFT 例へ合成。"""
    out: list[AssembledRecord] = []
    for ans in answers:
        if not ans.ok or not ans.cot_text.strip():
            continue
        out.append(assemble_one(ans, sampler, k, oracle_prob, seed))
    return out
