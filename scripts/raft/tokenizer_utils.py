"""
学習用トークナイザ（Swallow / Llama-3.1）でのトークン長計測。

トレーナー (継続事前学習/train.py::build_sft_dataset) は
    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tokenizer(text, truncation=True, max_length=2048)
を行う。長さフィルタはこれと同じ境界で測る必要があるため、生成側 (VLLM_MODEL) では
なく**学習側トークナイザ**を SWALLOW_TOKENIZER から遅延ロードして使う。

--dry-run 時は HF を読み込まず、文字数ベースの近似カウンタにフォールバックする。
"""

from __future__ import annotations

import logging
import os
from typing import Callable

logger = logging.getLogger(__name__)

_DEFAULT_TOKENIZER = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"

# 遅延シングルトン
_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        name = os.getenv("SWALLOW_TOKENIZER", _DEFAULT_TOKENIZER)
        logger.info(f"Loading tokenizer: {name}")
        _tokenizer = AutoTokenizer.from_pretrained(name)
    return _tokenizer


def count_chat_tokens(messages: list[dict]) -> int:
    """トレーナーと同一手順でチャットメッセージのトークン数を測る。"""
    tok = _get_tokenizer()
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return len(tok(text)["input_ids"])


# --- dry-run 用近似カウンタ（HF 不要） ---
# 日本語混在テキストの粗い目安: 概ね 1 トークン ≈ 2 文字。テンプレ分の固定オーバヘッドを加算。
_DRYRUN_CHARS_PER_TOKEN = 2.0
_DRYRUN_TEMPLATE_OVERHEAD = 8


def count_chat_tokens_dryrun(messages: list[dict]) -> int:
    """HF を使わずチャットテンプレ適用後の長さを近似する。"""
    chars = sum(len(m.get("content", "")) for m in messages)
    per_msg = _DRYRUN_TEMPLATE_OVERHEAD * len(messages)
    return int(chars / _DRYRUN_CHARS_PER_TOKEN) + per_msg


def make_token_counter(dry_run: bool = False) -> Callable[[list[dict]], int]:
    """長さ計測関数を返す（dry_run=True で HF 非依存の近似に切替）。"""
    return count_chat_tokens_dryrun if dry_run else count_chat_tokens
