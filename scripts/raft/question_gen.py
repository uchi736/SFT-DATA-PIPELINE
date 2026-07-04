"""
Phase1: チャンクから質問を生成（非同期 LLM）。

cpt-data-pipeline/scripts/augment/expand_qa_difficult.py の構造を踏襲:
  遅延 _get_agent() / asyncio.wait_for / Semaphore+gather / try/except でバッチを止めない。
--dry-run 時は LLM を呼ばず決定的なモック質問を返す。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # scripts/
from prompts import get_system_prompt, get_user_prompt, REPAIR_HINTS  # noqa: E402
from raft.schemas import Chunk, QuestionList, QuestionRecord, make_question_id  # noqa: E402
from raft import _genutil  # noqa: E402

# 1チャンクに渡すテキスト上限（プロンプト肥大を防ぐ）
_TEXT_LIMIT = 3000

_agents: dict = {}


def _get_agent(prompt_key: str = "raft_question"):
    if prompt_key not in _agents:
        from llm_factory import create_pydanticai_agent
        _agents[prompt_key] = create_pydanticai_agent(
            output_type=QuestionList,
            system_prompt=get_system_prompt(prompt_key),
        )
    return _agents[prompt_key]


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。．！？\?\!])\s*", text.strip())
    return [p.strip() for p in parts if p.strip()]


def mock_questions(chunk: Chunk, num_questions: int) -> list[str]:
    """dry-run 用の決定的モック質問（先頭文を素材化）。"""
    sents = _split_sentences(chunk.text) or [chunk.text[:40]]
    out = []
    for i in range(num_questions):
        s = sents[i % len(sents)]
        out.append(f"{s[:60]} について説明してください。")
    return out


async def _generate(text: str, num_questions: int, timeout: float,
                    prompt_key: str = "raft_question") -> list[str]:
    prompt = get_user_prompt(prompt_key, text=text[:_TEXT_LIMIT], num_questions=num_questions)
    settings = _genutil.guided_settings(QuestionList)

    def _valid(out: QuestionList) -> bool:
        return bool(out.questions) and all(q.strip() for q in out.questions)

    out = await _genutil.run_structured(
        _get_agent(prompt_key), prompt, timeout,
        model_settings=settings, validate=_valid, repair_hint=REPAIR_HINTS["json"],
    )
    if out is None:
        return []
    # 重複除去 + num_questions 件にクランプ
    seen, qs = set(), []
    for q in out.questions:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            qs.append(q)
    return qs[:num_questions]


async def run_phase1(
    golden_chunks: list[Chunk],
    num_questions: int,
    concurrency: int,
    timeout: float,
    dry_run: bool = False,
    done_chunk_ids: set | None = None,
    sink=None,
    prompt_key: str = "raft_question",
) -> list[QuestionRecord]:
    """ゴールデン候補チャンクごとにクエリ（質問/工程）を生成し QuestionRecord 列を返す。

    done_chunk_ids に含まれる chunk は再開時にスキップ。sink(dict) が与えられれば
    生成完了ごとに1レコードずつ追記（逐次チェックポイント, H1）。返り値は新規生成分のみ。
    prompt_key でタスク別プロンプト（qa=raft_question / regulation=regul_process）を選ぶ。
    """
    done = done_chunk_ids or set()
    todo = [c for c in golden_chunks if c.id not in done]
    new_records: list[QuestionRecord] = []

    def emit(chunk: Chunk, qs):
        for qi, q in enumerate(qs or []):
            r = QuestionRecord(
                question_id=make_question_id(chunk.id, qi),
                chunk_id=chunk.id, question=q,
                golden_text=chunk.text, golden_meta=chunk.meta,
            )
            new_records.append(r)
            if sink:
                sink(r.model_dump())

    if dry_run:
        for chunk in todo:
            emit(chunk, mock_questions(chunk, num_questions))
        return new_records

    await _genutil.run_incremental(
        todo,
        lambda c: _generate(c.text, num_questions, timeout, prompt_key),
        concurrency,
        lambda c, qs: emit(c, qs),
    )
    return new_records
