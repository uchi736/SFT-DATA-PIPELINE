"""
RAFT パイプラインのデータスキーマ（唯一の真実）。

- LLM 出力スキーマ（pydantic）: QuestionList, CoTAnswer
- 各フェーズの中間 JSONL レコード: QuestionRecord, AnswerRecord, AssembledRecord
- 最終出力: SFTRecord（トレーナー契約 {"messages": [...]} と一致）
- ID スキーム: chunk_id -> question_id -> example_id を安定リンク

全レコードは `.model_dump()` で dict 化し JSONL に1行ずつ書き出す。
中間レコードに golden_text を持ち回すことで resume 時のコーパス再結合を不要にする。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# コーパスのチャンク（入力 = cpt の _preprocessed.jsonl 互換）
# =============================================================================

@dataclass
class Chunk:
    """ドキュメントチャンク（ゴールデン/ディストラクタ共通）。"""
    id: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =============================================================================
# LLM 出力スキーマ（pydantic-ai の output_type / guided_json に使用）
# =============================================================================

class QuestionList(BaseModel):
    """Phase1: 1チャンクから生成した質問リスト。"""
    questions: list[str] = Field(description="文書だけで回答できる質問のリスト")


class CoTAnswer(BaseModel):
    """Phase2: RAFT スタイルの CoT 回答。"""
    reasoning: str = Field(
        description="##begin_quote##…##end_quote## の逐語引用を含む段階的な思考過程"
    )
    final_answer: str = Field(description="結論となる簡潔な最終回答")


# =============================================================================
# 中間 JSONL レコード（フェーズ間の受け渡し / checkpoint）
# =============================================================================

class QuestionRecord(BaseModel):
    """{stem}_questions.jsonl の1行。"""
    question_id: str
    chunk_id: str
    question: str
    golden_text: str
    golden_meta: dict[str, Any] = Field(default_factory=dict)


class AnswerRecord(BaseModel):
    """{stem}_answers.jsonl の1行。ok=False は生成失敗（export でスキップ）。"""
    question_id: str
    chunk_id: str
    question: str
    golden_text: str
    reasoning: str = ""
    final_answer: str = ""
    cot_text: str = ""
    ok: bool = False


class AssembledRecord(BaseModel):
    """{stem}_assembled.jsonl の1行（RAFT 合成済み・決定的）。"""
    example_id: str
    chunk_id: str
    question: str
    cot_text: str
    distractor_ids: list[str]
    oracle_included: bool
    doc_order: list[str]
    user_content: str
    assistant_content: str


class SFTRecord(BaseModel):
    """最終出力の1行（トレーナー契約）。"""
    messages: list[dict[str, str]]


# =============================================================================
# ID スキーム
# =============================================================================

def make_question_id(chunk_id: str, qi: int) -> str:
    """質問ID: chunk_id__q{index}"""
    return f"{chunk_id}__q{qi}"


def make_example_id(question_id: str) -> str:
    """例ID: 1質問=1例（oracleドロップアウトはフラグで表現）"""
    return question_id
