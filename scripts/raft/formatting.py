"""
RAFT の文脈整形・CoT回答整形・引用検証（LLM 非依存・決定的）。

- format_context: 複数ドキュメントを user メッセージへ整形（区切り + 末尾に質問）
- render_cot: reasoning + final_answer を assistant 出力へ整形
- has_valid_quotes: ##begin_quote##…##end_quote## が1組以上・正順で存在するか検証

引用マーカーは prompts.py と単一定義を共有する（循環 import 回避のため prompts から取得）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/ を import パスに追加（cpt 同様の sys.path 方式）
sys.path.insert(0, str(Path(__file__).parent.parent))
from prompts import QUOTE_OPEN, QUOTE_CLOSE  # noqa: E402

DOC_OPEN = "<DOCUMENT>"
DOC_CLOSE = "</DOCUMENT>"


def format_context(docs, question: str) -> str:
    """ドキュメント列 + 質問を user メッセージ文字列へ整形。

    Args:
        docs: Chunk のリスト（doc_order どおりの順序で渡す）
        question: 質問文
    """
    body = "\n".join(
        f"{DOC_OPEN}\n{doc.text.strip()}\n{DOC_CLOSE}" for doc in docs
    )
    return f"{body}\n\n質問: {question.strip()}"


def render_cot(reasoning: str, final_answer: str) -> str:
    """CoT 回答（assistant 出力）を整形。"""
    return f"{reasoning.strip()}\n\n最終的な回答: {final_answer.strip()}"


def count_quote_pairs(text: str) -> int:
    """正順に閉じている引用ペア数を数える（不正なネスト/逆順は無効）。"""
    pairs = 0
    idx = 0
    while True:
        b = text.find(QUOTE_OPEN, idx)
        if b == -1:
            break
        e = text.find(QUOTE_CLOSE, b + len(QUOTE_OPEN))
        if e == -1:
            break
        pairs += 1
        idx = e + len(QUOTE_CLOSE)
    return pairs


def has_valid_quotes(text: str) -> bool:
    """逐語引用が1組以上・正しく閉じているか。"""
    return count_quote_pairs(text) >= 1


def extract_quotes(text: str) -> list[str]:
    """##begin_quote##…##end_quote## の中身（前後空白除去）をすべて取り出す。"""
    out: list[str] = []
    idx = 0
    while True:
        b = text.find(QUOTE_OPEN, idx)
        if b == -1:
            break
        e = text.find(QUOTE_CLOSE, b + len(QUOTE_OPEN))
        if e == -1:
            break
        span = text[b + len(QUOTE_OPEN):e].strip()
        if span:
            out.append(span)
        idx = e + len(QUOTE_CLOSE)
    return out


def _norm_ws(s: str) -> str:
    """空白（改行・スペース等）をすべて除去して比較用に正規化。"""
    return "".join(s.split())


def quotes_grounded(text: str, source: str) -> bool:
    """全ての引用がコンテキスト本文の逐語（空白正規化後の部分文字列）であるか。

    引用が1つも無ければ False。RAFT ではパラフレーズした「引用」を学習ラベルに
    含めないため、answer_gen の検証に用いる。
    """
    quotes = extract_quotes(text)
    if not quotes:
        return False
    src = _norm_ws(source)
    return all(_norm_ws(q) in src for q in quotes)
