"""
RAFT SFT データ生成オーケストレータ（エントリポイント）。

使用例:
    # 本番（vLLM 起動済み）
    python scripts/build_raft.py data/input/corpus.jsonl -o data/output/sft_dataset.jsonl

    # vLLM 不要のスモーク（決定的モック）
    python scripts/build_raft.py tests/fixtures/tiny_corpus.jsonl -o data/output/smoke.jsonl \
        --dry-run --limit 4 --val-split 0.25

フロー: corpus → Phase1 質問 → Phase2 CoT回答 → Phase3 RAFT合成 → Phase4 エクスポート
Phase1/2 は LLM（checkpoint でスキップ可）。Phase3/4 は決定的・安価なので常に再実行。
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(Path(__file__).parent))  # scripts/

from prompts import SYSTEM_PROMPTS  # noqa: E402
from raft import checkpoint, corpus as corpus_mod  # noqa: E402
from raft import question_gen, answer_gen, assemble, export  # noqa: E402
from raft.schemas import QuestionRecord, AnswerRecord  # noqa: E402
from raft.tokenizer_utils import make_token_counter  # noqa: E402

CHECKPOINT_DIR = PROJECT_ROOT / "data" / ".checkpoint"
PH_QUESTIONS = "questions"
PH_ANSWERS = "answers"
PH_ASSEMBLED = "assembled"


def parse_args():
    p = argparse.ArgumentParser(
        description="RAFT SFT データ生成パイプライン",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="チャンク済みコーパス JSONL ({text,id,meta} または {text,source,doc})")
    p.add_argument("-o", "--output", required=True, help="出力 SFT JSONL パス")
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
                   help="literal 文字列 or SYSTEM_PROMPTS のキー")
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
    p.add_argument("--dry-run", action="store_true", help="LLM を呼ばず決定的モックで実行")
    p.add_argument("--keep-intermediate", action="store_true", help="中間 checkpoint を残す")
    return p.parse_args()


def _resolve_system_prompt(value):
    if value is None:
        return None
    return SYSTEM_PROMPTS.get(value, value)


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
    print(f"コーパス: 全{len(chunks)}件 / ゴールデン候補{len(goldens)}件 / "
          f"プロバイダ={'dry-run' if args.dry_run else 'llm'}")

    # --- プリフライト（本番のみ）---
    if not args.dry_run:
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
            )
        finally:
            ap.close()
        answers = [AnswerRecord(**d)
                   for d in _dedup_answers(checkpoint.read_phase(CHECKPOINT_DIR, stem, PH_ANSWERS))]
        checkpoint.mark_completed(CHECKPOINT_DIR, stem, PH_ANSWERS, sum(1 for a in answers if a.ok))
    ok_answers = sum(1 for a in answers if a.ok)
    print(f"  回答: 成功{ok_answers}/{len(answers)}件")

    # --- Phase3: RAFT合成（決定的・常に再実行）---
    print("Phase3: RAFT合成")
    assembled = assemble.run_phase3(
        answers, sampler, args.num_distractors, args.oracle_prob, args.seed,
    )
    oracle_in = sum(1 for a in assembled if a.oracle_included)
    print(f"  合成例: {len(assembled)}件 (oracle投入 {oracle_in}件 "
          f"≈ {oracle_in / max(1, len(assembled)):.2f})")
    if args.keep_intermediate:
        checkpoint.save_phase(CHECKPOINT_DIR, stem, PH_ASSEMBLED, [a.model_dump() for a in assembled])

    # --- Phase4: エクスポート ---
    print("Phase4: 長さフィルタ + エクスポート")
    counter = make_token_counter(dry_run=args.dry_run)
    records, stats = export.run_phase4(
        assembled, index, counter,
        max_len=args.max_len, oversize_policy=args.oversize_policy,
        min_distractors=args.min_distractors,
        system_prompt=_resolve_system_prompt(args.system_prompt),
    )
    paths = export.write_outputs(records, args.output, val_split=args.val_split, seed=args.seed)

    print(f"\n統計: {stats}")
    print(f"出力: {paths}")
    _eyeball(records)

    if not args.keep_intermediate and not args.resume:
        print(f"  [info] 中間チェックポイントを削除（{PH_QUESTIONS}/{PH_ANSWERS}）。"
              "保持/再開可能にするには --keep-intermediate を付けて実行してください。")
        checkpoint.reset(CHECKPOINT_DIR, stem)
    return 0


def main():
    args = parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
