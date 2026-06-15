"""
コーパス（チャンク済み JSONL）の読込・検証、決定的ディストラクタ抽出、質問の重複排除。

入力スキーマは柔軟に受ける:
  - cpt の _preprocessed.jsonl 互換 {"text", "id", "meta"}
  - cpt の corpus_clean.jsonl 互換 {"text", "source", "doc"}（id/meta 無し）
id が無い場合は**内容ハッシュ**で安定採番し（位置非依存＝--limit/並べ替えに強い）、
text/id 以外の補助フィールド（source/doc 等）は meta に畳み込む。
LLM 非依存・決定的。乱数は hashlib ベースでプロセス/プラットフォーム非依存に固定する
（Python 組込み hash() は PYTHONHASHSEED でぶれるため使わない）。
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import unicodedata
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))  # scripts/
from raft.schemas import Chunk, QuestionRecord  # noqa: E402

# 質問/ゴールデン源とするための最小文字数（短すぎるチャンクは除外）
MIN_CHUNK_CHARS = 200


# =============================================================================
# 決定的シード
# =============================================================================

def stable_seed(*parts) -> int:
    """文字列引数からプロセス非依存の決定的シード整数を作る。"""
    joined = "\x1f".join(str(p) for p in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


# =============================================================================
# テキスト正規化（任意 --clean-level）
# =============================================================================

def clean_text(text: str, level: str = "off") -> str:
    if level == "off":
        return text
    t = unicodedata.normalize("NFKC", text)
    if level in ("basic", "aggressive"):
        # 連続する空行を1つに圧縮
        lines = [ln.rstrip() for ln in t.splitlines()]
        out, blank = [], False
        for ln in lines:
            if ln == "":
                if not blank:
                    out.append("")
                blank = True
            else:
                out.append(ln)
                blank = False
        t = "\n".join(out).strip()
    if level == "aggressive":
        # 制御文字（タブ・改行以外）を除去
        t = "".join(ch for ch in t if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")
    return t


# =============================================================================
# 読込・検証
# =============================================================================

def split_text(text: str, max_chars: int) -> list[str]:
    """テキストを段落→文境界で max_chars 以下のセグメントへ貪欲分割。

    RAFT は複数文書を文脈に並べるため、CPT 向けの大きいチャンク（~1300字/~960tok）は
    max_len=2048 に収まらない。読み込み時に小さく割り直すための補助。
    """
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return [text] if text else []

    import re as _re
    segs: list[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            segs.append(buf.strip())
        buf = ""

    for para in [p for p in text.split("\n") if p.strip()]:
        if len(para) <= max_chars:
            if buf and len(buf) + 1 + len(para) > max_chars:
                flush()
            buf = (buf + "\n" + para) if buf else para
            continue
        # 段落が長い → 文分割
        flush()
        for s in _re.split(r"(?<=[。．！？!?])", para):
            if not s:
                continue
            if len(s) > max_chars:  # 1文が長すぎる → ハード分割
                flush()
                for i in range(0, len(s), max_chars):
                    segs.append(s[i:i + max_chars])
            else:
                if buf and len(buf) + len(s) > max_chars:
                    flush()
                buf += s
    flush()
    return segs or [text]


def load_corpus(
    path: str | Path,
    limit: Optional[int] = None,
    clean_level: str = "off",
    max_doc_chars: Optional[int] = None,
) -> list[Chunk]:
    """JSONL コーパスを Chunk リストとして読み込む（全件＝ディストラクタプール）。

    - text 欠落/空はスキップ
    - id があれば採用（重複はスキップ）、無ければ内容ハッシュで安定採番（source/doc は meta へ）
    - max_doc_chars 指定時は各チャンクを split_text で割り直し（RAFT 文脈を 2048 に収めるため）
    - limit は最終的な（分割後）チャンク数の上限
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"コーパスが見つかりません: {path}")

    chunks: list[Chunk] = []
    seen_ids: set[str] = set()
    n_autoid = 0
    n_dup = 0
    n_badjson = 0

    def add_piece(piece: str, rec: dict, meta_base: dict, multi: bool, part: int):
        nonlocal n_autoid, n_dup
        explicit = rec.get("id") or rec.get("chunk_id")
        if explicit and not multi:
            cid = str(explicit)
            if cid in seen_ids:
                n_dup += 1
                return
        else:
            n_autoid += 1
            base = str(rec.get("doc") or rec.get("source") or explicit or "chunk")
            h = hashlib.sha256(piece.encode("utf-8")).hexdigest()[:16]
            cid = f"{base}_{h}"
            if cid in seen_ids:  # 完全一致テキストがある場合のみ連番付与
                k = 1
                while f"{cid}-{k}" in seen_ids:
                    k += 1
                cid = f"{cid}-{k}"
        seen_ids.add(cid)
        meta = dict(meta_base)
        if multi:
            meta["part"] = part
        chunks.append(Chunk(id=cid, text=piece, meta=meta))

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(chunks) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_badjson += 1
                continue

            text = rec.get("text")
            if not text or not str(text).strip():
                continue
            text = clean_text(str(text), clean_level)

            # meta: 明示の meta を優先、無ければ text/id 以外（source/doc 等）を集約
            meta_base = rec.get("meta")
            if not isinstance(meta_base, dict):
                meta_base = {k: v for k, v in rec.items()
                             if k not in ("text", "id", "chunk_id", "meta")}

            pieces = split_text(text, max_doc_chars) if max_doc_chars else [text]
            multi = len(pieces) > 1
            for pi, piece in enumerate(pieces):
                if limit is not None and len(chunks) >= limit:
                    break
                if piece.strip():
                    add_piece(piece, rec, meta_base, multi, pi)

    # 警告は1行に集約（per-record の警告氾濫を防ぐ）
    if n_autoid:
        print(f"  [info] id欠落/分割 {n_autoid}件 → 内容ハッシュで採番（source/doc は meta へ保持）",
              file=sys.stderr)
    if n_dup:
        print(f"  [warn] id重複スキップ {n_dup}件", file=sys.stderr)
    if n_badjson:
        print(f"  [warn] JSON解析失敗 {n_badjson}行", file=sys.stderr)

    return chunks


def index_corpus(chunks: list[Chunk]) -> dict[str, Chunk]:
    return {c.id: c for c in chunks}


def golden_candidates(chunks: list[Chunk], min_chars: int = MIN_CHUNK_CHARS) -> list[Chunk]:
    """質問/ゴールデン源として十分な長さのチャンクのみ（短いものはディストラクタ専用）。"""
    return [c for c in chunks if len(c.text) >= min_chars]


# =============================================================================
# 決定的ディストラクタサンプラ
# =============================================================================

def make_distractor_sampler(chunks: list[Chunk], seed: int) -> Callable[..., list[Chunk]]:
    """ゴールデンを除いたプールから k 件のディストラクタを決定的に抽出する関数を返す。

    sampler(golden_id, k, salt="") -> list[Chunk]
    同じ (seed, golden_id, salt) は常に同じ結果。プールが k 未満なら min(k, len(pool)) にクランプ。
    """
    ids = [c.id for c in chunks]
    by_id = {c.id: c for c in chunks}

    def sample(golden_id: str, k: int, salt: str = "") -> list[Chunk]:
        pool = [i for i in ids if i != golden_id]
        n = min(k, len(pool))
        if n <= 0:
            return []
        rng = random.Random(stable_seed(seed, golden_id, salt))
        chosen = rng.sample(pool, n)
        return [by_id[i] for i in chosen]

    return sample


# =============================================================================
# 質問の重複排除
# =============================================================================

def _normalize_question(q: str) -> str:
    return unicodedata.normalize("NFKC", q).strip().lower()


def dedup_questions(
    records: list[QuestionRecord], global_dedup: bool = False
) -> list[QuestionRecord]:
    """正規化した質問文で重複排除。既定はチャンク内、global_dedup=True で全体。"""
    seen: set = set()
    out: list[QuestionRecord] = []
    for r in records:
        norm = _normalize_question(r.question)
        key = norm if global_dedup else (r.chunk_id, norm)
        if not norm or key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out
