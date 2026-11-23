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


CONTEXT_FORMATS = ("document", "bullets")
BULLETS_HEADER = "参考資料:"


def format_context(docs, question: str, query_label: str = "質問",
                   style: str = "document", instruction: str | None = None) -> str:
    """ドキュメント列 + クエリを user メッセージ文字列へ整形。

    Args:
        docs: Chunk のリスト（doc_order どおりの順序で渡す）
        question: クエリ本文（qa=質問 / regulation=工程 など）
        query_label: 末尾クエリの見出し（既定「質問」、regulation では「工程」）
        style: "document" = <DOCUMENT> タグ区切り（既定）
               "bullets"  = 「参考資料:」見出し + 箇条書き（RAG の推論時の整形に合わせる用途）
        instruction: user の先頭に置く指示文（system を使わない土台向け）。None で付けない
    """
    if style == "bullets":
        body = BULLETS_HEADER + "\n" + "\n".join(f"- {doc.text.strip()}" for doc in docs)
        out = f"{body}\n{query_label}: {question.strip()}"
    elif style == "document":
        body = "\n".join(
            f"{DOC_OPEN}\n{doc.text.strip()}\n{DOC_CLOSE}" for doc in docs
        )
        out = f"{body}\n\n{query_label}: {question.strip()}"
    else:
        raise ValueError(f"未知の context format: {style}")
    if instruction and instruction.strip():
        out = f"{instruction.strip()}\n\n{out}"
    return out


def render_doc_list(docs) -> str:
    """文書列を番号なしの箇条書きにする（拒否文の作文・判定の指示に渡す見え方）。"""
    return "\n".join(f"- {doc.text.strip()}" for doc in docs)


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


def norm_text(s: str) -> str:
    """NFKC 正規化 + 空白除去（全角/半角・改行の違いを無視した照合用）。"""
    import unicodedata
    return "".join(unicodedata.normalize("NFKC", s).split())


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


QUOTE_MIN_CHARS = 10      # これより短い引用は照合に使わない（偶然の一致を避ける）


def answer_quotes(cot_text: str) -> list[str]:
    """CoT 回答の逐語引用（NFKC 正規化・空白除去済み。短いものは除く）。"""
    qs = [norm_text(q) for q in extract_quotes(cot_text)]
    return [q for q in qs if len(q) >= QUOTE_MIN_CHARS]


def carries_answer(doc_text: str, golden_text: str, quotes: list[str]) -> bool:
    """文書が正解の内容を持っているか（機械照合）。

    本文が正解の文書と同じ（ID 違いの重複）か、回答の引用をそのまま含む場合に True。
    ディストラクタや差し替え候補から外す判定に使う。言い換えで同じ内容を持つ文書は拾えない。
    """
    n = norm_text(doc_text)
    return n == norm_text(golden_text) or any(q in n for q in quotes)


def carries_all_quotes(doc_text: str, golden_text: str, quotes: list[str]) -> bool:
    """文書だけで回答の根拠がそろうか（機械照合）: 本文が正解の文書と同じか、回答の引用をすべて含む。

    carries_answer（引用を 1 つでも含む）より厳しい。短い決まり文句の引用 1 つだけが一致した文書を
    「同じ内容」と数えないために使う。
    """
    n = norm_text(doc_text)
    return n == norm_text(golden_text) or (bool(quotes) and all(q in n for q in quotes))
