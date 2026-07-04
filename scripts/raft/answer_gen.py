"""
Phase2: ゴールデン（オラクル）文書に基づく RAFT スタイル CoT 回答生成（非同期 LLM）。

回答はオラクル文書のみを根拠に作る（ディストラクタは Phase3 で付与）。RAFT 論文どおり、
reasoning に ##begin_quote##…##end_quote## の逐語引用を最低1組含めることを検証し、
欠落時は修復リトライ。--dry-run 時は決定的モックを返す。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # scripts/
from prompts import (  # noqa: E402
    get_system_prompt, get_user_prompt, REPAIR_HINTS, QUOTE_OPEN, QUOTE_CLOSE,
)
from raft.schemas import AnswerRecord, CoTAnswer, QuestionRecord  # noqa: E402
from raft import _genutil, formatting  # noqa: E402

# 回答プロンプトに渡すオラクル本文の上限
_CONTEXT_LIMIT = 3000

_agents: dict = {}


def _get_agent(prompt_key: str = "raft_cot_answer"):
    if prompt_key not in _agents:
        from llm_factory import create_pydanticai_agent
        _agents[prompt_key] = create_pydanticai_agent(
            output_type=CoTAnswer,
            system_prompt=get_system_prompt(prompt_key),
        )
    return _agents[prompt_key]


def mock_cot(question: str, golden_text: str) -> CoTAnswer:
    """dry-run 用の決定的モック CoT（オラクル先頭を引用）。"""
    snippet = golden_text.strip()[:40]
    reasoning = (
        f"質問「{question}」に答えるため、コンテキストの該当箇所を確認する。"
        f"{QUOTE_OPEN} {snippet} {QUOTE_CLOSE} とあることから、関連内容を読み取れる。"
    )
    final = f"{snippet} に基づくと、上記の通りです。"
    return CoTAnswer(reasoning=reasoning, final_answer=final)


async def _generate(question: str, golden_text: str, timeout: float,
                    prompt_key: str = "raft_cot_answer") -> CoTAnswer | None:
    ctx = golden_text[:_CONTEXT_LIMIT]
    prompt = get_user_prompt(prompt_key, context=ctx, question=question)
    settings = _genutil.guided_settings(CoTAnswer)
    repair_hint = REPAIR_HINTS["quote"] + "\n" + REPAIR_HINTS["json"]

    def _valid(out: CoTAnswer) -> bool:
        # 最終回答あり + 引用ペアが閉じている + 引用が逐語（コンテキストの部分文字列, N4）
        return (
            bool(out.final_answer.strip())
            and formatting.has_valid_quotes(out.reasoning)
            and formatting.quotes_grounded(out.reasoning, ctx)
        )

    return await _genutil.run_structured(
        _get_agent(prompt_key), prompt, timeout,
        model_settings=settings, validate=_valid, repair_hint=repair_hint,
        max_attempts=3,
    )


def _to_record(q: QuestionRecord, cot: CoTAnswer | None) -> AnswerRecord:
    if cot is None:
        return AnswerRecord(
            question_id=q.question_id, chunk_id=q.chunk_id,
            question=q.question, golden_text=q.golden_text, ok=False,
        )
    cot_text = formatting.render_cot(cot.reasoning, cot.final_answer)
    return AnswerRecord(
        question_id=q.question_id, chunk_id=q.chunk_id,
        question=q.question, golden_text=q.golden_text,
        reasoning=cot.reasoning, final_answer=cot.final_answer,
        cot_text=cot_text, ok=True,
    )


async def run_phase2(
    questions: list[QuestionRecord],
    concurrency: int,
    timeout: float,
    dry_run: bool = False,
    done_question_ids: set | None = None,
    sink=None,
    prompt_key: str = "raft_cot_answer",
) -> list[AnswerRecord]:
    """クエリ（質問/工程）ごとに CoT 回答/判定を生成し AnswerRecord 列を返す。

    done_question_ids に含まれるものは再開時にスキップ。sink(dict) が与えられれば
    生成完了ごとに1レコードずつ追記（逐次チェックポイント, H1）。返り値は新規生成分のみ。
    prompt_key でタスク別プロンプト（qa=raft_cot_answer / regulation=regul_judge）を選ぶ。
    """
    done = done_question_ids or set()
    todo = [q for q in questions if q.question_id not in done]
    new_records: list[AnswerRecord] = []

    def emit(q: QuestionRecord, cot):
        r = _to_record(q, cot)
        new_records.append(r)
        if sink:
            sink(r.model_dump())

    if dry_run:
        for q in todo:
            emit(q, mock_cot(q.question, q.golden_text))
        return new_records

    await _genutil.run_incremental(
        todo,
        lambda q: _generate(q.question, q.golden_text, timeout, prompt_key),
        concurrency,
        lambda q, cot: emit(q, cot),
    )
    return new_records
