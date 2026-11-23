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
    # --absent-mode refuse のときだけ True。ゴールデンは doc_order に「差し替える位置の印」として
    # 残してあり、このままでは書き出さない（拒否例フェーズで別の文書に差し替え、答えを拒否文にする）。
    refuse: bool = False
    # --distractor-mode retrieved のときの記録。retrieved では、検索が正解の文書を取り逃した例は
    # ゴールデンを doc_order に持たない（refuse=True でも差し替える位置の印は無い）。
    distractor_mode: str = "random"
    golden_rank: int = -1              # 検索の最終順位（0 始まり）。候補に入らなければ -1
    n_mixed: int = 0                   # 無作為の文書に置き換えた枠の数（--mix-random）
    dropped_ids: list[str] = Field(default_factory=list)   # 答えを含むため外した検索結果


class RetrievalRecord(BaseModel):
    """{stem}_retrieved.jsonl の1行（質問ごとの検索結果）。"""
    question_id: str
    chunk_id: str                      # 正解の文書
    ranked: list[str] = Field(default_factory=list)      # 最終順位の文書 ID 列（top_k 件）
    scores: list[float] = Field(default_factory=list)    # リランカーの関連度（リランカーなしは空）
    golden_rank: int = -1              # ranked の中の正解の文書の位置。無ければ -1
    golden_bm25_rank: int = -1         # 融合前の BM25 の順位（POOL 件の中。無ければ -1）
    golden_dense_rank: int = -1        # 融合前の埋め込みの順位（同上）
    config_key: str = ""               # 検索の設定の指紋（resume 時に設定変更を検出）


class QuestionReview(BaseModel):
    """{stem}_question_review.jsonl の1行（質問の判定）。

    verdict: accept（自動採用）/ review（要確認）/ reject（自動除外）
    reasons: 当たった条件。{"code", "label", "detail"} の列
    """
    question_id: str
    chunk_id: str
    question: str
    final_answer: str = ""
    verdict: str = "accept"
    reasons: list[dict[str, str]] = Field(default_factory=list)
    override: str = ""                 # 人の判定で上書きした場合、上書き前の判定


class RefusalRecord(BaseModel):
    """{stem}_refusals.jsonl の1行（拒否例。1 例 1 行、逐次追記）。

    kind:
      absent = 正解の文書が文脈に無い質問（回答例は作らない）。oracle ドロップアウトで抜いたものと、
               検索が取り逃したもの（--distractor-mode retrieved。slot=-1 で差し替えなし）がある
      twin   = 回答例と同じ質問・同じ文書の並びで、正解の文書だけを差し替えた対
    status: ok / skip（候補が無い・全候補で答えが導ける・拒否文が検証に落ちた）/ error（LLM 呼び出し失敗、resume で再試行）
    """
    example_id: str
    question_id: str
    kind: str
    chunk_id: str                      # 抜いた正解の文書
    question: str
    gold: str = ""                     # 模範解答（最終回答）。漏れの検査と判定に使う
    layout_key: str = ""               # 差し替え前の並びの指紋（resume 時に設定変更を検出）
    n_docs: int = 0
    slot: int = -1                     # 差し替えた位置（差し替えなしは -1）
    doc_order: list[str] = Field(default_factory=list)   # 差し替え後の文書 ID 列
    substitute_id: str = ""
    cand_rank: int = -1                # 何番目の候補を採ったか（0 始まり）
    opener: int = -1
    refusal: str = ""
    fails: list[str] = Field(default_factory=list)       # 拒否文が検証に落ちた理由（再生成の履歴）
    status: str = "skip"
    reason: str = ""


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


def make_twin_id(question_id: str) -> str:
    """双子（同じ質問の拒否版）の例ID。回答例の ID とは別にする。"""
    return f"{question_id}__twin"
