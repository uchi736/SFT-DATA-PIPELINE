from raft import formatting
from raft.formatting import DOC_OPEN, DOC_CLOSE
from raft.schemas import Chunk
from prompts import QUOTE_OPEN, QUOTE_CLOSE


def _docs():
    return [
        Chunk(id="x", text="本文1。"),
        Chunk(id="y", text="本文2。"),
    ]


def test_format_context_has_delimiters_and_trailing_question():
    out = formatting.format_context(_docs(), "これは何ですか")
    assert out.count(DOC_OPEN) == 2
    assert out.count(DOC_CLOSE) == 2
    assert out.rstrip().endswith("質問: これは何ですか")


def test_has_valid_quotes_positive():
    text = f"根拠は {QUOTE_OPEN} 引用本文 {QUOTE_CLOSE} である。"
    assert formatting.has_valid_quotes(text)
    assert formatting.count_quote_pairs(text) == 1


def test_has_valid_quotes_missing_close():
    text = f"根拠は {QUOTE_OPEN} 引用本文 のみ。"
    assert not formatting.has_valid_quotes(text)


def test_has_valid_quotes_reversed_order():
    text = f"{QUOTE_CLOSE} 逆順 {QUOTE_OPEN}"
    assert not formatting.has_valid_quotes(text)


def test_render_cot_contains_final_answer_label():
    out = formatting.render_cot("推論。", "結論。")
    assert "最終的な回答: 結論。" in out


SOURCE = "ガス軸受は気体の粘度が低いため高速回転時の摩擦損失が小さい。"


def test_extract_quotes():
    text = f"理由は {QUOTE_OPEN} 気体の粘度が低い {QUOTE_CLOSE} ためです。"
    assert formatting.extract_quotes(text) == ["気体の粘度が低い"]


def test_quotes_grounded_verbatim_passes():
    text = f"根拠: {QUOTE_OPEN} 高速回転時の摩擦損失が小さい {QUOTE_CLOSE}"
    assert formatting.quotes_grounded(text, SOURCE)


def test_quotes_grounded_whitespace_normalized():
    # 引用内の空白/改行差は許容（空白正規化後の部分文字列判定）
    text = f"{QUOTE_OPEN} 高速回転時の\n摩擦損失が小さい {QUOTE_CLOSE}"
    assert formatting.quotes_grounded(text, SOURCE)


def test_quotes_grounded_paraphrase_fails():
    # パラフレーズ（逐語でない）は弾く
    text = f"{QUOTE_OPEN} 摩擦がとても小さいらしい {QUOTE_CLOSE}"
    assert not formatting.quotes_grounded(text, SOURCE)


def test_quotes_grounded_no_quotes_fails():
    assert not formatting.quotes_grounded("引用なしの本文。", SOURCE)
