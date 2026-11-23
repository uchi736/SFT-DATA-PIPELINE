"""
RAFT SFT データ生成オーケストレータ（エントリポイント）。

使用例:
    # 本番（vLLM 起動済み）
    python scripts/build_raft.py data/input/corpus.jsonl -o data/output/sft_dataset.jsonl

    # vLLM 不要のスモーク（決定的モック）
    python scripts/build_raft.py tests/fixtures/tiny_corpus.jsonl -o data/output/smoke.jsonl \
        --dry-run --limit 4 --val-split 0.25

    # 拒否例つき（正解の文書を抜いた例は断る答えにし、同じ質問の双子も作る）
    python scripts/build_raft.py data/input/corpus.jsonl -o data/output/sft_refuse.jsonl \
        --absent-mode refuse --twin-ratio 0.4 --context-format bullets \
        --user-instruction rag_refuse --val-split 0.1 --keep-intermediate

    # ディストラクタを検索の上位から取り、質問を判定してから使う（推論時の形に合わせる）
    python scripts/build_raft.py data/input/corpus.jsonl -o data/output/sft_hardneg.jsonl \
        --max-doc-chars 450 --distractor-mode retrieved --question-check on \
        --absent-mode refuse --twin-ratio 0.25 --context-format bullets \
        --user-instruction rag_refuse --val-split 0.1 --keep-intermediate

フロー: corpus → Phase1 質問 → Phase2 CoT回答 → (検索) → (質問の判定) → Phase3 RAFT合成
        → (拒否例) → Phase4 エクスポート
Phase1/2 と拒否例は LLM（checkpoint でスキップ可）。検索は埋め込みとリランカーの API を呼ぶ
（checkpoint でスキップ可）。質問の判定と Phase3/4 は決定的・安価なので常に再実行。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(Path(__file__).parent))  # scripts/

from prompts import (  # noqa: E402
    TASKS, get_sft_system_prompt, get_sft_user_instruction, get_task,
)
from raft import checkpoint, corpus as corpus_mod  # noqa: E402
from raft import question_gen, answer_gen, assemble, export, formatting, refusal  # noqa: E402
from raft import question_check, retrieval  # noqa: E402
from raft.schemas import QuestionRecord, AnswerRecord, RefusalRecord, RetrievalRecord  # noqa: E402
from raft.tokenizer_utils import make_token_counter  # noqa: E402

CHECKPOINT_DIR = PROJECT_ROOT / "data" / ".checkpoint"
PH_QUESTIONS = "questions"
PH_ANSWERS = "answers"
PH_ASSEMBLED = "assembled"
PH_REFUSALS = "refusals"
PH_RETRIEVED = "retrieved"
CACHE_DIR = PROJECT_ROOT / "data" / ".cache"


def parse_args():
    p = argparse.ArgumentParser(
        description="RAFT SFT データ生成パイプライン",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="チャンク済みコーパス JSONL ({text,id,meta} または {text,source,doc})")
    p.add_argument("-o", "--output", required=True, help="出力 SFT JSONL パス")
    p.add_argument("--task", choices=list(TASKS.keys()), default="qa",
                   help="qa=RAFT質問応答 / regulation=工程→該当規定の判定")
    p.add_argument("--questions-per-chunk", type=int, default=3)
    p.add_argument("--num-distractors", type=int, default=4, help="k: ディストラクタ数")
    p.add_argument("--oracle-prob", type=float, default=0.8, help="P: ゴールデン投入確率")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-len", type=int, default=2048, help="Swallow トークンの上限")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout", type=float, default=300.0,
                   help="1 LLM 呼び出しの秒数（クライアント側 VLLM_TIMEOUT=300 と整合）")
    p.add_argument("--resume", action="store_true", help="checkpoint から再開")
    p.add_argument("--val-split", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None, help="読み込むチャンク数の上限")
    p.add_argument("--system-prompt", default=None,
                   help="出力に注入する system。プリセット名 {raft, rag_strict, assistant_ja} か任意の文字列")
    p.add_argument("--system-dropout", type=float, default=0.0,
                   help="この割合の例を system 無しで出力（推論時のsystemブレに頑健化、0〜1）")
    p.add_argument("--oversize-policy", choices=["drop", "reduce"], default="reduce")
    p.add_argument("--min-distractors", type=int, default=1)
    p.add_argument("--min-chunk-chars", type=int, default=corpus_mod.MIN_CHUNK_CHARS,
                   help="質問/ゴールデン源とする最小文字数")
    p.add_argument("--max-doc-chars", type=int, default=None,
                   help="各チャンクをこの文字数以下に再分割（大きいCPTチャンクをRAFT文脈に収める。例 450）")
    p.add_argument("--max-goldens", type=int, default=None,
                   help="質問を作るゴールデンチャンク数の上限（全件からseed付きサンプル。小規模データ用）")
    p.add_argument("--clean-level", choices=["off", "basic", "aggressive"], default="off")
    p.add_argument("--global-dedup", action="store_true", help="質問重複排除を全体で行う")
    p.add_argument("--absent-mode", choices=list(assemble.ABSENT_MODES), default="answer",
                   help="ゴールデンを抜いた例の答え。answer=CoT回答のまま（RAFT論文どおり）/ "
                        "refuse=拒否文にし、文書数を回答例とそろえる（qa のみ）")
    p.add_argument("--twin-ratio", type=float, default=0.0,
                   help="回答例のうち、同じ質問でゴールデンだけ差し替えた拒否例（双子）も作る割合（0〜1、qa のみ）")
    p.add_argument("--context-format", choices=list(formatting.CONTEXT_FORMATS), default="document",
                   help="文脈の書式。document=<DOCUMENT>タグ / bullets=「参考資料:」+箇条書き")
    p.add_argument("--user-instruction", default=None,
                   help="user の先頭に置く指示文。プリセット名 {rag_refuse} か任意の文字列")
    p.add_argument("--distractor-mode", choices=list(assemble.DISTRACTOR_MODES), default="random",
                   help="ディストラクタの選び方。random=コーパス全体から無作為（RAFT 論文の公式実装と同じ）/ "
                        "retrieved=検索の上位から取る。正解の文書が上位に入らない質問がゴールデン抜きの例になり、"
                        "--oracle-prob は使わない")
    p.add_argument("--retrieval-mode", choices=list(retrieval.MODES), default="hybrid",
                   help="retrieved の検索方式。hybrid=BM25 と埋め込みの順位融合")
    p.add_argument("--no-rerank", action="store_true", help="retrieved でリランカーを通さない")
    p.add_argument("--retrieval-top-k", type=int, default=20,
                   help="retrieved でリランカーに渡す候補数（リランカーなしでは検索結果の件数）")
    p.add_argument("--retrieval-concurrency", type=int, default=4, help="リランカーへの同時要求数")
    p.add_argument("--mix-random", type=float, default=0.2,
                   help="retrieved で、正解以外の各枠を無作為の文書に置き換える確率（0〜1。0 で検索結果だけ）")
    p.add_argument("--doc-order", choices=list(assemble.DOC_ORDERS), default="shuffle",
                   help="retrieved での文書の並び。shuffle=入れ替える / rank=検索の順位のまま"
                        "（正解の文書が 1 番目に偏る。manifest の golden_position_dist で確認できる）")
    p.add_argument("--question-check", choices=["off", "on"], default="off",
                   help="on=質問を 自動採用/要確認/自動除外 に判定し、理由つきの確認用ファイルを書く。"
                        "自動除外は学習例に使わない")
    p.add_argument("--review-policy", choices=["hold", "keep"], default="hold",
                   help="要確認の質問の扱い。hold=学習例に使わない / keep=使う")
    p.add_argument("--question-overrides", default=None,
                   help="人の判定で上書きする JSONL（1 行 {question_id, verdict}。確認用ファイルを編集したものでよい）")
    p.add_argument("--dry-run", action="store_true",
                   help="LLM を呼ばず決定的モックで実行（retrieved の検索は BM25 だけになる）")
    p.add_argument("--keep-intermediate", action="store_true", help="中間 checkpoint を残す")
    return p.parse_args()


def _resolve_system_prompt(value):
    # 名前なら SFT_SYSTEM_PROMPTS から展開、それ以外は文字列そのまま
    return get_sft_system_prompt(value)


def _dedup_answers(dicts: list[dict]) -> list[dict]:
    """answer dict を question_id で重複排除（ok=True を優先）。

    resume で失敗(ok=False)を再生成すると同一 question_id が二重に追記されるため、
    成功版を優先して1件に畳む。
    """
    by_qid: dict = {}
    for d in dicts:
        cur = by_qid.get(d["question_id"])
        if cur is None or (not cur.get("ok") and d.get("ok")):
            by_qid[d["question_id"]] = d
    return list(by_qid.values())


def _md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _median(xs: list[int]) -> int:
    return sorted(xs)[len(xs) // 2] if xs else 0


def _kind_stats(records, metas, idx) -> dict:
    """種類ごとの件数と形（長さ・文書数）。拒否例と回答例が形で見分けられないかの確認用。"""
    out: dict = {}
    for kind in sorted({metas[i]["kind"] for i in idx}):
        sel = [i for i in idx if metas[i]["kind"] == kind]
        users = [next(m["content"] for m in records[i].messages if m["role"] == "user") for i in sel]
        assts = [next(m["content"] for m in records[i].messages if m["role"] == "assistant") for i in sel]
        nd = Counter(metas[i]["n_docs"] for i in sel)
        out[kind] = {"n": len(sel), "user_len_median": _median([len(u) for u in users]),
                     "assistant_len_median": _median([len(a) for a in assts]),
                     "n_docs_dist": {str(k): nd[k] for k in sorted(nd)}}
    n_ref = sum(v["n"] for k, v in out.items() if k in ("absent", "twin"))
    return {"n": len(idx), "refusal_ratio": round(n_ref / max(1, len(idx)), 3), "kinds": out}


def _write_manifest(args, output_path, chunks, goldens, records, metas, stats,
                    train_idx, val_idx, refusal_info, extra=None) -> str:
    """生成の記録（使ったモデル・引数・入力の md5・種類ごとの件数と形）を出力の隣に書く。"""
    from llm_factory import get_model_name, get_provider
    out = Path(output_path)
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "provider": "dry-run" if args.dry_run else get_provider(),
        "model": None if args.dry_run else get_model_name(),
        "args": {k: v for k, v in sorted(vars(args).items()) if k not in ("input", "output")},
        "input": {"file": Path(args.input).name, "md5": _md5(args.input),
                  "chunks": len(chunks), "goldens": len(goldens)},
        "export_stats": stats,
        "splits": {"all": _kind_stats(records, metas, list(range(len(records))))},
    }
    if val_idx:
        manifest["splits"]["train"] = _kind_stats(records, metas, train_idx)
        manifest["splits"]["val"] = _kind_stats(records, metas, val_idx)
    if refusal_info:
        manifest["refusal"] = refusal_info
    manifest.update(extra or {})
    path = out.with_name(out.stem + "_manifest.json")
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return str(path)


def _doc_of(chunk) -> str:
    """チャンクの元の文書名（meta.doc。無ければ ID から内容ハッシュを除いたもの）。"""
    d = (chunk.meta or {}).get("doc") if chunk is not None else None
    return str(d) if d else re.sub(r"_[0-9a-f]{16}(-\d+)?$", "", chunk.id if chunk is not None else "")


def _context_stats(assembled, index) -> dict:
    """文脈の作られ方の要約（manifest 用）: ゴールデンの有無・順位・ディストラクタの出どころ。"""
    n = len(assembled)
    if not n:
        return {"n": 0}
    n_dis = sum(len(a.distractor_ids) for a in assembled)
    same = sum(1 for a in assembled for d in a.distractor_ids
               if _doc_of(index.get(d)) == _doc_of(index.get(a.chunk_id)))
    out = {
        "n": n,
        "oracle_included": sum(1 for a in assembled if a.oracle_included),
        "oracle_ratio": round(sum(1 for a in assembled if a.oracle_included) / n, 3),
        "n_docs_dist": {str(k): v for k, v in sorted(Counter(len(a.doc_order) for a in assembled).items())},
        "distractors": n_dis,
        "distractors_same_doc_as_golden": round(same / max(1, n_dis), 3),
    }
    if any(a.distractor_mode == "retrieved" for a in assembled):
        pos = Counter(a.doc_order.index(a.chunk_id) for a in assembled if a.oracle_included)
        out.update({
            "golden_position_dist": {str(k): v for k, v in sorted(pos.items())},
            "mixed_random_slots": sum(a.n_mixed for a in assembled),
            "dropped_answer_carrying_docs": sum(len(a.dropped_ids) for a in assembled),
            "examples_with_dropped_docs": sum(1 for a in assembled if a.dropped_ids),
        })
    return out


def _preflight() -> str | None:
    """本番(非dry-run)前に vLLM 疎通とモデル一致を確認。問題があればメッセージ、無ければ None。

    エンドポイント誤り/未起動/モデル不一致を ~6200 回叩く前に1回で検知する（N3）。
    """
    from llm_factory import get_provider
    if get_provider() != "vllm":
        return None  # azure 等はここでは省略（実呼び出しで判明）
    import os
    import httpx
    ep = os.getenv("VLLM_ENDPOINT", "http://localhost:8000/v1")
    model = os.getenv("VLLM_MODEL", "")
    try:
        r = httpx.get(
            ep.rstrip("/") + "/models",
            timeout=10,
            headers={"Authorization": "Bearer " + os.getenv("VLLM_API_KEY", "EMPTY")},
        )
    except Exception as e:
        return f"vLLM へ接続できません ({ep}): {type(e).__name__}. VLLM_ENDPOINT を確認してください。"
    if r.status_code != 200:
        return f"vLLM /models が HTTP {r.status_code} を返しました ({ep})。"
    ids = [m.get("id") for m in r.json().get("data", [])]
    if model and model not in ids:
        return f"VLLM_MODEL='{model}' が配信モデル {ids} に一致しません。"
    print(f"  preflight OK: {ep} / {model}")
    return None


def _eyeball(records, n=3):
    if not records:
        return
    rng = random.Random(0)
    sample = rng.sample(records, min(n, len(records)))
    print("\n--- サンプル ---")
    for i, r in enumerate(sample, 1):
        msgs = r.messages
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        asst = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        print(f"[例{i}] user(先頭120字): {user[:120].replace(chr(10), ' ')}")
        print(f"       assistant(先頭120字): {asst[:120].replace(chr(10), ' ')}")


async def run(args):
    stem = Path(args.output).stem
    task = get_task(args.task)
    with_refusals = args.absent_mode == "refuse" or args.twin_ratio > 0
    if with_refusals and args.task != "qa":
        print("エラー: --absent-mode refuse / --twin-ratio は --task qa でのみ使えます", file=sys.stderr)
        return 1
    if not 0.0 <= args.twin_ratio <= 1.0:
        print("エラー: --twin-ratio は 0〜1 で指定してください", file=sys.stderr)
        return 1
    retrieved_mode = args.distractor_mode == "retrieved"
    if not 0.0 <= args.mix_random <= 1.0:
        print("エラー: --mix-random は 0〜1 で指定してください", file=sys.stderr)
        return 1
    user_instruction = get_sft_user_instruction(args.user_instruction)
    if not args.resume:
        checkpoint.reset(CHECKPOINT_DIR, stem)

    # --- コーパス ---
    chunks = corpus_mod.load_corpus(
        args.input, limit=args.limit, clean_level=args.clean_level,
        max_doc_chars=args.max_doc_chars,
    )
    if not chunks:
        print("エラー: 有効なチャンクがありません", file=sys.stderr)
        return 1
    goldens = corpus_mod.golden_candidates(chunks, min_chars=args.min_chunk_chars)
    if not goldens:
        print("エラー: ゴールデン候補（十分な長さのチャンク）がありません", file=sys.stderr)
        return 1
    if args.max_goldens and len(goldens) > args.max_goldens:
        rng = random.Random(args.seed)
        goldens = sorted(rng.sample(goldens, args.max_goldens), key=lambda c: c.id)
        print(f"  ゴールデンを {args.max_goldens} 件にサンプル（seed={args.seed}、ディストラクタは全{len(chunks)}件から）")
    index = corpus_mod.index_corpus(chunks)
    sampler = corpus_mod.make_distractor_sampler(chunks, args.seed)
    print(f"タスク: {args.task} / コーパス: 全{len(chunks)}件 / ゴールデン候補{len(goldens)}件 / "
          f"プロバイダ={'dry-run' if args.dry_run else 'llm'}")

    # --- プリフライト（実際に生成する場合のみ。再エクスポートだけなら vLLM 不要）---
    needs_llm = not args.dry_run and not (
        args.resume
        and checkpoint.is_completed(CHECKPOINT_DIR, stem, PH_QUESTIONS)
        and checkpoint.is_completed(CHECKPOINT_DIR, stem, PH_ANSWERS)
    )
    if needs_llm:
        err = _preflight()
        if err:
            print(f"エラー(preflight): {err}", file=sys.stderr)
            return 1

    # --- Phase1: 質問生成（逐次チェックポイント / resume 対応）---
    if args.resume and checkpoint.is_completed(CHECKPOINT_DIR, stem, PH_QUESTIONS):
        raw_q = checkpoint.read_phase(CHECKPOINT_DIR, stem, PH_QUESTIONS)
        questions = corpus_mod.dedup_questions(
            [QuestionRecord(**d) for d in raw_q], global_dedup=args.global_dedup)
    else:
        done = (checkpoint.existing_keys(CHECKPOINT_DIR, stem, PH_QUESTIONS, "chunk_id")
                if args.resume else set())
        todo_n = sum(1 for c in goldens if c.id not in done)
        print(f"Phase1: 質問生成 (対象 {todo_n}/{len(goldens)}チャンク x {args.questions_per_chunk}"
              + (f", 再開: 既存{len(done)})" if done else ")"))
        ap = checkpoint.Appender(CHECKPOINT_DIR, stem, PH_QUESTIONS, resume=args.resume)
        try:
            await question_gen.run_phase1(
                goldens, args.questions_per_chunk, args.concurrency, args.timeout,
                dry_run=args.dry_run, done_chunk_ids=done, sink=ap.write,
                prompt_key=task["p1"],
            )
        finally:
            ap.close()
        raw_q = checkpoint.read_phase(CHECKPOINT_DIR, stem, PH_QUESTIONS)
        questions = corpus_mod.dedup_questions(
            [QuestionRecord(**d) for d in raw_q], global_dedup=args.global_dedup)
        checkpoint.mark_completed(CHECKPOINT_DIR, stem, PH_QUESTIONS, len(questions))
    print(f"  質問: {len(questions)}件")
    if not questions:
        print("エラー: 質問が0件です", file=sys.stderr)
        return 1

    # --- Phase2: CoT回答生成（逐次チェックポイント / resume 対応）---
    if args.resume and checkpoint.is_completed(CHECKPOINT_DIR, stem, PH_ANSWERS):
        answers = [AnswerRecord(**d)
                   for d in _dedup_answers(checkpoint.read_phase(CHECKPOINT_DIR, stem, PH_ANSWERS))]
    else:
        # resume では ok=True のみ「完了」扱い（失敗 ok=False は再試行する）
        done = set()
        if args.resume:
            for d in checkpoint.read_phase(CHECKPOINT_DIR, stem, PH_ANSWERS):
                if d.get("ok"):
                    done.add(d["question_id"])
        todo_n = sum(1 for q in questions if q.question_id not in done)
        print(f"Phase2: CoT回答生成 (対象 {todo_n}/{len(questions)}件"
              + (f", 再開: 成功済み{len(done)})" if done else ")"))
        ap = checkpoint.Appender(CHECKPOINT_DIR, stem, PH_ANSWERS, resume=args.resume)
        try:
            await answer_gen.run_phase2(
                questions, args.concurrency, args.timeout,
                dry_run=args.dry_run, done_question_ids=done, sink=ap.write,
                prompt_key=task["p2"],
            )
        finally:
            ap.close()
        answers = [AnswerRecord(**d)
                   for d in _dedup_answers(checkpoint.read_phase(CHECKPOINT_DIR, stem, PH_ANSWERS))]
        checkpoint.mark_completed(CHECKPOINT_DIR, stem, PH_ANSWERS, sum(1 for a in answers if a.ok))
    ok_answers = sum(1 for a in answers if a.ok)
    print(f"  回答: 成功{ok_answers}/{len(answers)}件")

    extra_manifest: dict = {}

    # --- 検索（--distractor-mode retrieved。逐次チェックポイント / resume 対応）---
    retrieved: dict[str, RetrievalRecord] = {}
    ranked_by_qid = None
    if retrieved_mode:
        retriever = retrieval.Retriever(
            chunks, mode=args.retrieval_mode, use_rerank=not args.no_rerank,
            top_k=max(args.retrieval_top_k, args.num_distractors + 1),
            dry_run=args.dry_run, cache_dir=CACHE_DIR)
        done_ret = {}
        if args.resume:
            for d in checkpoint.read_phase(CHECKPOINT_DIR, stem, PH_RETRIEVED):
                done_ret[d["question_id"]] = RetrievalRecord(**d)
        targets = [a for a in answers if a.ok and a.cot_text.strip()]
        n_reuse = sum(1 for a in targets if a.question_id in done_ret
                      and done_ret[a.question_id].config_key == retriever.config_key)
        print(f"検索: {retriever.describe()} (対象 {len(targets) - n_reuse}/{len(targets)}件"
              + (f", 再開: 既存{n_reuse})" if n_reuse else ")"))
        if not args.dry_run and n_reuse < len(targets):
            err = retrieval.preflight(retriever.mode, retriever.use_rerank)
            if err:
                print(f"エラー(検索の事前確認): {err}", file=sys.stderr)
                return 1
        ap = checkpoint.Appender(CHECKPOINT_DIR, stem, PH_RETRIEVED, resume=args.resume)
        try:
            retrieved = retrieval.run_retrieval(
                targets, retriever, concurrency=args.retrieval_concurrency,
                done=done_ret, sink=ap.write)
        finally:
            ap.close()
        checkpoint.mark_completed(CHECKPOINT_DIR, stem, PH_RETRIEVED, len(retrieved))
        ranked_by_qid = {qid: r.ranked for qid, r in retrieved.items()}
        ret_stats = retrieval.summarize(retrieved, args.num_distractors + 1)
        print(f"  検索の結果: {ret_stats}")
        extra_manifest["retrieval"] = {**retriever.describe(), **ret_stats}

    fmt = dict(query_label=task["query_label"], context_format=args.context_format,
               user_instruction=user_instruction)
    p3 = dict(absent_mode=args.absent_mode, distractor_mode=args.distractor_mode,
              ranked_by_qid=ranked_by_qid, index=index, mix_random=args.mix_random,
              doc_order=args.doc_order, **fmt)

    # --- 質問の判定（--question-check on。決定的・常に再実行）---
    if args.question_check == "on":
        carried = None
        if retrieved_mode:
            # 検索の上位に、回答の引用をすべて含む別の文書があるか（注記として残す）
            carried = {}
            for a in answers:
                if not a.ok or a.question_id not in ranked_by_qid:
                    continue
                quotes = formatting.answer_quotes(a.cot_text)
                ids = [cid for cid in ranked_by_qid[a.question_id]
                       if cid != a.chunk_id and cid in index
                       and formatting.carries_all_quotes(index[cid].text, a.golden_text, quotes)]
                if ids:
                    carried[a.question_id] = ids
        overrides = (question_check.load_overrides(args.question_overrides)
                     if args.question_overrides else None)
        reviews = question_check.check_all(answers, retrieved or None, carried, overrides)
        qc_stats = question_check.summarize(reviews)
        review_path = Path(args.output).with_name(stem + "_question_review.jsonl")
        review_path.parent.mkdir(parents=True, exist_ok=True)
        with open(review_path, "w", encoding="utf-8") as f:
            for r in reviews:
                f.write(json.dumps(r.model_dump(), ensure_ascii=False) + "\n")
        usable = question_check.usable_ids(reviews, args.review_policy)
        n_before = sum(1 for a in answers if a.ok and a.cot_text.strip())
        answers = [a for a in answers if a.question_id in usable]
        print(f"質問の判定: {qc_stats['verdicts']} 理由 {qc_stats['reasons']}")
        print(f"  学習例に使う質問: {len(answers)}/{n_before}件 (要確認の扱い={args.review_policy}) → {review_path}")
        extra_manifest["question_check"] = {**qc_stats, "review_policy": args.review_policy,
                                            "used": len(answers), "review_file": review_path.name}
        if not answers:
            print("エラー: 判定の後に使える質問が0件です", file=sys.stderr)
            return 1

    # --- Phase3: RAFT合成（決定的・常に再実行）---
    print("Phase3: RAFT合成" + ("（ディストラクタ=検索の上位。--oracle-prob は使わない）"
                                 if retrieved_mode else ""))
    assembled = assemble.run_phase3(
        answers, sampler, args.num_distractors, args.oracle_prob, args.seed, **p3)
    oracle_in = sum(1 for a in assembled if a.oracle_included)
    print(f"  合成例: {len(assembled)}件 (oracle投入 {oracle_in}件 "
          f"≈ {oracle_in / max(1, len(assembled)):.2f})")
    ctx_stats = _context_stats(assembled, index)
    extra_manifest["context"] = ctx_stats
    if retrieved_mode:
        print(f"  文脈: {ctx_stats}")
    if args.keep_intermediate:
        checkpoint.save_phase(CHECKPOINT_DIR, stem, PH_ASSEMBLED, [a.model_dump() for a in assembled])

    counter = make_token_counter(dry_run=args.dry_run)
    fit = dict(max_len=args.max_len, oversize_policy=args.oversize_policy,
               min_distractors=args.min_distractors,
               system_prompt=_resolve_system_prompt(args.system_prompt),
               system_dropout=args.system_dropout, **fmt)

    # --- 拒否例（--absent-mode refuse / --twin-ratio。逐次チェックポイント / resume 対応）---
    refusals: list[RefusalRecord] = []
    refusal_info = None
    if with_refusals:
        plan = refusal.plan_tasks(
            assembled, {a.question_id: a for a in answers}, index, counter,
            twin_ratio=args.twin_ratio, seed=args.seed, **fit,
        )
        n_twin = sum(plan.twin_quotas.values())
        print(f"拒否例: oracle なし {len(plan.absent)}件 + 双子 {n_twin}件 "
              f"(回答例 {plan.n_answers}件の {args.twin_ratio:.2f}、文書数ごとの枠 {plan.twin_quotas})")
        done = {}
        if args.resume:
            for d in checkpoint.read_phase(CHECKPOINT_DIR, stem, PH_REFUSALS):
                done[d["example_id"]] = RefusalRecord(**d)   # 同じ例は後の行が優先
            if done:
                print(f"  再開: 既存 {len(done)}件 (ok {sum(1 for r in done.values() if r.status == 'ok')})")
        if not args.dry_run and not needs_llm and refusal.has_pending(plan, done):
            # 質問・回答は checkpoint で足りたが、拒否例はこれから作る → ここで疎通を確かめる
            err = _preflight()
            if err:
                print(f"エラー(preflight): {err}", file=sys.stderr)
                return 1
        ap = checkpoint.Appender(CHECKPOINT_DIR, stem, PH_REFUSALS, resume=args.resume)
        try:
            results = await refusal.run_refusal_phase(
                plan, refusal.NearPool(chunks), index, args.concurrency,
                dry_run=args.dry_run, done=done, sink=ap.write,
                ranked_by_qid=ranked_by_qid,
            )
        finally:
            ap.close()
        refusals = refusal.select_refusals(plan, results)
        planned = {t.example_id for t in plan.absent} | {
            t.example_id for ts in plan.twin_buckets.values() for t in ts}
        tried = [r for k, r in results.items() if k in planned]
        n_err = sum(1 for r in tried if r.status == "error")
        checkpoint.mark_completed(CHECKPOINT_DIR, stem, PH_REFUSALS, len(refusals))
        refusal_info = {
            "absent": {"planned": len(plan.absent),
                       "ok": sum(1 for r in refusals if r.kind == "absent")},
            "twin": {"quota": n_twin, "ok": sum(1 for r in refusals if r.kind == "twin"),
                     "quotas_by_n_docs": {str(k): v for k, v in sorted(plan.twin_quotas.items())}},
            "not_ok_reasons": dict(Counter(
                (r.reason.split(":")[0] or r.status) for r in tried if r.status != "ok")),
            "substitute_rank_dist": dict(Counter(str(r.cand_rank) for r in refusals)),
            "opener_dist": dict(Counter(str(r.opener) for r in refusals)),
            "slot_dist": dict(Counter(str(r.slot) for r in refusals)),
            "write_retries": sum(len(r.fails) for r in refusals),
            "skipped_oversize": plan.skipped_oversize,
        }
        print(f"  採用: oracle なし {refusal_info['absent']['ok']}/{len(plan.absent)}件, "
              f"双子 {refusal_info['twin']['ok']}/{n_twin}件  不採用の理由: {refusal_info['not_ok_reasons']}")
        if n_err:
            print(f"  [warn] LLM 呼び出しに失敗した例が {n_err}件あります。--resume で再試行できます。",
                  file=sys.stderr)

    # --- Phase4: エクスポート ---
    print("Phase4: 長さフィルタ + エクスポート")
    records, metas, stats = export.run_phase4_mixed(
        assembled, refusals, index, counter, seed=args.seed, shuffle=with_refusals, **fit,
    )
    # 拒否例を混ぜる場合は質問単位で分ける（回答例と双子が train と val に分かれないように）
    groups = [m["group"] for m in metas] if with_refusals else None
    paths = export.write_outputs(records, args.output, val_split=args.val_split, seed=args.seed,
                                 groups=groups)
    train_idx, val_idx = export.split_indices(len(records), args.val_split, args.seed, groups)
    paths["manifest"] = _write_manifest(args, args.output, chunks, goldens, records, metas, stats,
                                        train_idx, val_idx, refusal_info, extra_manifest)

    print(f"\n統計: {stats}")
    print(f"出力: {paths}")
    if with_refusals:
        for name, st in json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))["splits"].items():
            kinds = ", ".join(f"{k} {v['n']}" for k, v in st["kinds"].items())
            print(f"  {name}: {st['n']}件 ({kinds}) 拒否の割合 {st['refusal_ratio']}")
    _eyeball(records)

    if not args.keep_intermediate and not args.resume:
        print(f"  [info] 中間チェックポイントを削除（{PH_QUESTIONS}/{PH_ANSWERS}"
              + (f"/{PH_REFUSALS}" if with_refusals else "") + "）。"
              "保持/再開可能にするには --keep-intermediate を付けて実行してください。")
        checkpoint.reset(CHECKPOINT_DIR, stem)
    return 0


def main():
    args = parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
