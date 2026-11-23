"""
検索フェーズ: 質問ごとに、推論時と同じ形の検索でコーパスを引き、順位つきの候補を返す。

  BM25（文字 2-gram + 3-gram）と埋め込みで各 POOL 件 → 順位融合（RRF）で top_k 件 → リランカーで並べ替え

--distractor-mode retrieved のときだけ使う。ここで得た順位から、Phase3 がディストラクタ
（検索の上位にあるが正解ではない文書）を取り、正解の文書が上位に入らなかった質問を
「ゴールデン抜き」の例にする。

検索ロジック:
BM25 は k1=1.5, b=0.75、順位融合は 1/(61+順位) の和、各 POOL=50 件。

接続先は環境変数で指定する（.env）。いずれも OpenAI 互換 API:
  EMBED_ENDPOINT / EMBED_MODEL / EMBED_QUERY_PREFIX / EMBED_DOC_PREFIX
  RERANK_ENDPOINT / RERANK_MODEL / RERANK_QUERY_PREFIX / RERANK_DOC_PREFIX

失敗を黙って握りつぶさない: リランカーの呼び出しが失敗したら例外を上げる（順位を恒等のまま返すと、
モデル名の誤りに気付かないまま全件が「検索の順位」として使われる）。

--dry-run 時はネットワークを使わず BM25 だけで順位を決める（決定的）。
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))  # scripts/
from raft.schemas import AnswerRecord, Chunk, RetrievalRecord  # noqa: E402

POOL = 50            # BM25・埋め込みのそれぞれから融合に回す件数
RRF_K = 60           # 順位融合の定数（1/(RRF_K + 1 + 順位)）
EMBED_BATCH = 32
MODES = ("hybrid", "dense", "bm25")


# =============================================================================
# 接続先
# =============================================================================

def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def embed_config() -> dict:
    return {
        "endpoint": _env("EMBED_ENDPOINT", "http://localhost:8003/v1").rstrip("/"),
        "model": _env("EMBED_MODEL", "cl-nagoya/ruri-v3-310m"),
        "query_prefix": _env("EMBED_QUERY_PREFIX", "検索クエリ: "),
        "doc_prefix": _env("EMBED_DOC_PREFIX", "検索文書: "),
    }


def rerank_config() -> dict:
    return {
        "endpoint": _env("RERANK_ENDPOINT", "http://localhost:8006/v1").rstrip("/"),
        "model": _env("RERANK_MODEL", "cl-nagoya/ruri-v3-reranker-310m"),
        "query_prefix": _env("RERANK_QUERY_PREFIX", ""),
        "doc_prefix": _env("RERANK_DOC_PREFIX", ""),
    }


def _timeout() -> float:
    return float(os.getenv("RETRIEVAL_TIMEOUT", "180"))


def _post(url: str, body: dict) -> dict:
    import httpx
    r = httpx.post(url, json=body, timeout=_timeout(),
                   headers={"Authorization": "Bearer " + os.getenv("RETRIEVAL_API_KEY", "EMPTY")})
    r.raise_for_status()
    return r.json()


def _get(url: str) -> dict:
    import httpx
    r = httpx.get(url, timeout=10,
                  headers={"Authorization": "Bearer " + os.getenv("RETRIEVAL_API_KEY", "EMPTY")})
    r.raise_for_status()
    return r.json()


# =============================================================================
# BM25（文字 n-gram。形態素解析なしで日本語の専門用語を拾う）
# =============================================================================

def char_ngrams(s: str, ns=(2, 3)) -> list[str]:
    s = re.sub(r"\s+", "", s).lower()
    return [s[i:i + n] for n in ns for i in range(len(s) - n + 1)]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(texts)
        grams = [Counter(char_ngrams(t)) for t in texts]
        self.doc_len = [sum(c.values()) for c in grams]
        self.avgdl = sum(self.doc_len) / max(1, self.N)
        self.inverted: dict[str, dict[int, int]] = defaultdict(dict)
        for di, c in enumerate(grams):
            for g, tf in c.items():
                self.inverted[g][di] = tf
        self.idf = {g: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5))
                    for g, p in self.inverted.items()}

    def scores(self, query: str) -> dict[int, float]:
        sc: dict[int, float] = defaultdict(float)
        for g in set(char_ngrams(query)):
            posting = self.inverted.get(g)
            if not posting:
                continue
            idf = self.idf[g]
            for di, tf in posting.items():
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[di] / self.avgdl)
                sc[di] += idf * tf * (self.k1 + 1) / denom
        return sc

    def ranked(self, query: str, n: int) -> list[int]:
        # 同点は文書の番号順（決定的）
        return [di for di, _ in sorted(self.scores(query).items(), key=lambda x: (-x[1], x[0]))[:n]]


def rrf(rank_lists: list[list[int]], top_k: int) -> list[int]:
    """順位融合。複数の順位表を 1/(RRF_K + 1 + 順位) の和でまとめ、上位 top_k 件を返す。"""
    fused: dict[int, float] = defaultdict(float)
    for ranked in rank_lists:
        for r, di in enumerate(ranked):
            fused[di] += 1.0 / (RRF_K + 1 + r)
    return [di for di, _ in sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:top_k]]


# =============================================================================
# 埋め込み・リランカー（HTTP）
# =============================================================================

def embed_texts(texts: list[str], is_query: bool):
    """正規化済みの埋め込み行列（float32）を返す。"""
    import numpy as np
    cfg = embed_config()
    pfx = cfg["query_prefix"] if is_query else cfg["doc_prefix"]
    vecs: list = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = [pfx + t for t in texts[i:i + EMBED_BATCH]]
        d = _post(cfg["endpoint"] + "/embeddings", {"model": cfg["model"], "input": batch})
        data = sorted(d["data"], key=lambda e: e.get("index", 0))
        if len(data) != len(batch):
            raise RuntimeError(f"埋め込みの件数が合いません: 送信 {len(batch)} / 受信 {len(data)}")
        vecs += [e["embedding"] for e in data]
    v = np.asarray(vecs, dtype=np.float32)
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)


def rerank_scores(query: str, docs: list[str]) -> list[float]:
    """docs と同じ並びの関連度を返す。失敗時は例外（恒等順に逃げない）。"""
    cfg = rerank_config()
    d = _post(cfg["endpoint"] + "/rerank", {
        "model": cfg["model"], "query": cfg["query_prefix"] + query,
        "documents": [cfg["doc_prefix"] + t for t in docs],
    })
    results = d.get("results") or []
    if len(results) != len(docs):
        raise RuntimeError(f"リランカーの結果の件数が合いません: 送信 {len(docs)} / 受信 {len(results)}")
    out = [0.0] * len(docs)
    for r in results:
        out[int(r["index"])] = float(r["relevance_score"])
    return out


def preflight(mode: str, use_rerank: bool) -> Optional[str]:
    """検索の接続先とモデル名を確かめる。問題があればメッセージ、無ければ None。

    リランカーは、明らかに関連する文書と無関係の文書を 1 組渡し、関連する方が上に来ることも確かめる。
    """
    if mode in ("hybrid", "dense"):
        cfg = embed_config()
        try:
            ids = [m.get("id") for m in _get(cfg["endpoint"] + "/models").get("data", [])]
        except Exception as e:
            return f"埋め込みの接続先に届きません ({cfg['endpoint']}): {type(e).__name__}"
        if cfg["model"] not in ids:
            return f"EMBED_MODEL='{cfg['model']}' が配信モデル {ids} に一致しません。"
    if use_rerank:
        cfg = rerank_config()
        try:
            ids = [m.get("id") for m in _get(cfg["endpoint"] + "/models").get("data", [])]
        except Exception as e:
            return f"リランカーの接続先に届きません ({cfg['endpoint']}): {type(e).__name__}"
        if cfg["model"] not in ids:
            return f"RERANK_MODEL='{cfg['model']}' が配信モデル {ids} に一致しません。"
        try:
            s = rerank_scores("ボイラの定期自主検査はいつ行うか",
                              ["ボイラの定期自主検査は、一年以内ごとに一回、定期に行わなければならない。",
                               "本日の昼食は、食堂でカレーライスを提供する。"])
        except Exception as e:
            return f"リランカーの試し打ちに失敗しました: {type(e).__name__}: {e}"[:300]
        if not s[0] > s[1]:
            return f"リランカーの試し打ちで、関連する文書が上に来ません（{s}）。モデルと接頭辞を確認してください。"
    print(f"  検索の事前確認 OK: mode={mode} rerank={use_rerank}")
    return None


# =============================================================================
# 検索器
# =============================================================================

def _corpus_fingerprint(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.id.encode("utf-8"))
        h.update(b"\x1f")
        h.update(hashlib.sha256(c.text.encode("utf-8")).digest())
    return h.hexdigest()[:16]


class Retriever:
    """コーパス 1 つ分の検索器。search(question) で順位つきの文書 ID を返す。"""

    def __init__(self, chunks: list[Chunk], mode: str = "hybrid", use_rerank: bool = True,
                 top_k: int = 20, dry_run: bool = False, cache_dir: Optional[Path] = None):
        if mode not in MODES:
            raise ValueError(f"未知の検索方式: {mode}")
        if dry_run:
            mode, use_rerank = "bm25", False     # ネットワークを使わない
        self.chunks = chunks
        self.ids = [c.id for c in chunks]
        self.pos = {cid: i for i, cid in enumerate(self.ids)}
        self.mode, self.use_rerank, self.top_k = mode, use_rerank, max(1, top_k)
        self.dry_run = dry_run
        self.cache_dir = cache_dir
        self.fingerprint = _corpus_fingerprint(chunks)
        self.bm25 = BM25([c.text for c in chunks]) if mode in ("hybrid", "bm25") else None
        self._doc_emb = None
        self._q_emb: dict[str, object] = {}

    # ---- 設定の指紋（checkpoint を使い回せるかの判定に使う）----
    @property
    def config_key(self) -> str:
        parts = [self.mode, str(self.use_rerank), str(self.top_k), str(POOL), self.fingerprint]
        if self.mode in ("hybrid", "dense"):
            e = embed_config()
            parts += [e["model"], e["query_prefix"], e["doc_prefix"]]
        if self.use_rerank:
            r = rerank_config()
            parts += [r["model"], r["query_prefix"], r["doc_prefix"]]
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]

    def describe(self) -> dict:
        d = {"mode": self.mode, "rerank": self.use_rerank, "top_k": self.top_k, "pool": POOL,
             "config_key": self.config_key}
        if self.mode in ("hybrid", "dense"):
            d["embed_model"] = embed_config()["model"]
        if self.use_rerank:
            d["rerank_model"] = rerank_config()["model"]
        return d

    # ---- 埋め込み ----
    def _doc_embeddings(self):
        if self._doc_emb is not None:
            return self._doc_emb
        import numpy as np
        e = embed_config()
        key = hashlib.sha256("\x1f".join(
            [self.fingerprint, e["model"], e["doc_prefix"]]).encode("utf-8")).hexdigest()[:16]
        path = (self.cache_dir / f"docemb_{key}.npy") if self.cache_dir else None
        if path and path.exists():
            emb = np.load(path)
            if emb.shape[0] == len(self.chunks):
                self._doc_emb = emb
                return emb
        print(f"  コーパスの埋め込みを作成: {len(self.chunks)}件 ({e['model']})")
        emb = embed_texts([c.text for c in self.chunks], is_query=False)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, emb)
        self._doc_emb = emb
        return emb

    def prepare(self, questions: list[str]) -> None:
        """質問の埋め込みをまとめて作っておく（search を並行に呼ぶ前に 1 回）。"""
        if self.mode not in ("hybrid", "dense"):
            return
        self._doc_embeddings()
        todo = [q for q in dict.fromkeys(questions) if q not in self._q_emb]
        if not todo:
            return
        vecs = embed_texts(todo, is_query=True)
        for q, v in zip(todo, vecs):
            self._q_emb[q] = v

    def _dense_ranked(self, question: str, n: int) -> list[int]:
        import numpy as np
        if question not in self._q_emb:
            self.prepare([question])
        sims = self._doc_embeddings() @ self._q_emb[question]
        return [int(i) for i in np.argsort(-sims, kind="stable")[:n]]

    # ---- 検索 ----
    def search(self, question: str) -> dict:
        """戻り値: {"ranked": [文書 ID], "scores": [float], "bm25": [index], "dense": [index]}

        ranked は最終の順位（リランカーありならその順）。scores はリランカーの関連度
        （リランカーなしでは空）。bm25 / dense は融合前の各 POOL 件（文書の番号）。
        """
        bm = self.bm25.ranked(question, POOL) if self.bm25 is not None else []
        de = self._dense_ranked(question, POOL) if self.mode in ("hybrid", "dense") else []
        if self.mode == "bm25":
            cand = bm[:self.top_k]
        elif self.mode == "dense":
            cand = de[:self.top_k]
        else:
            cand = rrf([de, bm], self.top_k)
        scores: list[float] = []
        if self.use_rerank and cand:
            s = rerank_scores(question, [self.chunks[i].text for i in cand])
            order = sorted(range(len(cand)), key=lambda j: (-s[j], j))
            cand = [cand[j] for j in order]
            scores = [s[j] for j in order]
        return {"ranked": [self.ids[i] for i in cand], "scores": scores, "bm25": bm, "dense": de}

    def record(self, ans: AnswerRecord) -> RetrievalRecord:
        res = self.search(ans.question)
        gi = self.pos.get(ans.chunk_id)

        def rank_in(lst) -> int:
            return lst.index(gi) if gi is not None and gi in lst else -1

        ranked = res["ranked"]
        return RetrievalRecord(
            question_id=ans.question_id, chunk_id=ans.chunk_id,
            ranked=ranked, scores=[round(s, 6) for s in res["scores"]],
            golden_rank=ranked.index(ans.chunk_id) if ans.chunk_id in ranked else -1,
            golden_bm25_rank=rank_in(res["bm25"]), golden_dense_rank=rank_in(res["dense"]),
            config_key=self.config_key,
        )


# =============================================================================
# フェーズ実行
# =============================================================================

def run_retrieval(
    answers: list[AnswerRecord],
    retriever: Retriever,
    concurrency: int = 4,
    done: Optional[dict[str, RetrievalRecord]] = None,
    sink: Optional[Callable[[dict], None]] = None,
) -> dict[str, RetrievalRecord]:
    """成功した回答の質問ごとに検索し、question_id → RetrievalRecord を返す。

    done のうち設定の指紋（config_key）が同じものは使い回す。sink(dict) があれば 1 件ごとに追記。
    検索の失敗は例外としてそのまま上げる（一部だけ順位が無い状態で先へ進まない）。
    """
    key = retriever.config_key
    records = {qid: r for qid, r in (done or {}).items() if r.config_key == key}
    todo = [a for a in answers if a.ok and a.cot_text.strip() and a.question_id not in records]
    if not todo:
        return records
    retriever.prepare([a.question for a in todo])
    workers = 1 if retriever.dry_run or not retriever.use_rerank else max(1, concurrency)
    if workers == 1:
        for a in todo:
            rec = retriever.record(a)
            records[a.question_id] = rec
            if sink:
                sink(rec.model_dump())
        return records
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(retriever.record, a): a for a in todo}
        for fut in as_completed(futs):
            rec = fut.result()            # 例外はここで上がる
            records[rec.question_id] = rec
            if sink:
                sink(rec.model_dump())
    return records


def summarize(records: dict[str, RetrievalRecord], n_ctx: int) -> dict:
    """検索の結果の要約（manifest 用）。n_ctx = 文脈に入れる文書数。"""
    n = len(records)
    if not n:
        return {"n": 0}
    ranks = [r.golden_rank for r in records.values()]

    def hit(k: int) -> float:
        return round(sum(1 for g in ranks if 0 <= g < k) / n, 3)

    return {
        "n": n,
        "golden_in_candidates": round(sum(1 for g in ranks if g >= 0) / n, 3),
        "hit@1": hit(1), "hit@3": hit(3), f"hit@{n_ctx}": hit(n_ctx),
        "golden_in_bm25_pool": round(sum(1 for r in records.values() if r.golden_bm25_rank >= 0) / n, 3),
        "golden_in_dense_pool": round(sum(1 for r in records.values() if r.golden_dense_rank >= 0) / n, 3),
    }
