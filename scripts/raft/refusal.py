"""
拒否例フェーズ: 正解の文書が文脈に無い例の答えを「資料に無いので答えない」文にする（非同期 LLM）。

RAFT 論文どおりの合成では、確率 (1 - oracle_prob) で正解の文書を文脈から抜くが、答えは正解の文書を
見て書いた引用付き回答のまま残る。「資料に無ければ断る」ことを教えたい場合、これらの例は逆の手本になる。
このフェーズは次の 2 種類の拒否例を作る。

  absent : oracle ドロップアウトで正解の文書を抜いた質問（--absent-mode refuse）。回答例は作らない
  twin   : 回答例と同じ質問・同じ文書の並びで、正解の文書だけを別の文書に差し替えた対（--twin-ratio）。
           同じ質問が「正解の文書あり → 回答」「正解の文書なし → 拒否」の 2 行で入る

形で見分けられないようにする（中身を読まずに拒否/回答を当てる近道を残さない）ための決まり:
  1. 文書数・並び・書式は回答例と同じ。正解の文書の位置に、質問に字面が近く長さも近い別の文書を入れる
  2. 差し替え後の文書群から模範解答が導けないことを LLM で判定する（導けたら次の候補へ。
     同じ内容が複数の文書にあるコーパスでは必須）
  3. 拒否文は回答例と同じ骨格（主題の確認 → 資料の確認 → 最終的な回答:）で LLM に書かせ、機械検証する
     （長さ・引用マーカー無し・最終回答行が拒否・模範解答の漏れ無し など）。3 回落ちたら捨てる
  4. 双子は文書数ごとに同じ割合で作る（特定の文書数だけ拒否が多い、という手掛かりを残さない）
  5. 拒否文が作れなかった例は捨てる（元の引用付き回答には戻さない）

--distractor-mode retrieved のときの違い:
  - absent は「検索が正解の文書を取り逃した質問」になる。差し替えはせず、検索の上位の文書群を
    そのまま見せる。その文書群から模範解答が導ける場合は捨てる（理由 answerable_from_retrieved）
  - twin の差し替え候補は、字面の近さではなく検索の順位で次点だった文書から引く

--dry-run 時は LLM を呼ばず決定的なモックを返す。
"""

from __future__ import annotations

import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))  # scripts/
from prompts import (  # noqa: E402
    get_user_prompt, REFUSAL_OPENERS, REFUSE_MARKERS,
)
from raft.schemas import (  # noqa: E402
    AnswerRecord, AssembledRecord, Chunk, RefusalRecord, make_twin_id,
)
from raft.corpus import stable_seed  # noqa: E402
from raft import _genutil, export, formatting  # noqa: E402

MAX_CANDIDATES = 3        # 差し替え候補の数（答えが導けたら次の候補へ）
MAX_WRITE_TRIES = 3       # 拒否文の再生成回数
LEN_WINDOW = 0.3          # 差し替え候補の長さは正解の文書の ±30% を優先
QUOTE_MIN_CHARS = 10      # これより短い引用は照合に使わない（偶然の一致を避ける）
REFUSAL_MIN_CHARS = 200
REFUSAL_MAX_CHARS = 800
FINAL_PREFIX = "最終的な回答"
_FINAL_RE = re.compile(r"最終的な回答[:：]\s*(.+)", re.DOTALL)


# =============================================================================
# 文字列の照合
# =============================================================================

def bigrams(s: str) -> set[str]:
    s = formatting.norm_text(s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def final_answer_of(text: str) -> str:
    """「最終的な回答:」以降の最初の段落を 1 行にして返す。無ければ最後の行。"""
    t = text.strip()
    m = _FINAL_RE.search(t)
    if m:
        return " ".join(m.group(1).strip().split("\n\n")[0].split())
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def is_refusal(text: str) -> bool:
    """最終回答が拒否か。途中で「記載がありません」と述べつつ最後は答える出力を拒否と数えないよう、
    最終回答行だけを見る（取れない場合のみ全文）。"""
    target = final_answer_of(text) or text
    return any(m in target for m in REFUSE_MARKERS)


def looks_broken(text: str) -> bool:
    return text.count("##begin_quote") + text.count("##end_quote") > 8


def leaks_gold(text: str, gold: str, question: str, n: int = QUOTE_MIN_CHARS) -> bool:
    """模範解答の n 文字連（質問文にも現れるものは除く）が拒否文に現れたら漏れとみなす。"""
    t, g, q = formatting.norm_text(text), formatting.norm_text(gold), formatting.norm_text(question)
    grams = {g[i:i + n] for i in range(len(g) - n + 1)} - {q[i:i + n] for i in range(len(q) - n + 1)}
    return any(gr in t for gr in grams)


def validate_refusal(text: str, gold: str, question: str) -> Optional[str]:
    """拒否文の機械検証。通れば None、落ちれば理由。"""
    if not (REFUSAL_MIN_CHARS <= len(text) <= REFUSAL_MAX_CHARS):
        return f"length {len(text)}"
    if "##begin_quote" in text or "##end_quote" in text:
        return "quote_marker"
    if FINAL_PREFIX not in text:
        return "no_final"
    if not is_refusal(text):
        return "final_not_refusal"
    if looks_broken(text):
        return "broken"
    if re.search(r"模範回答|骨格", text) or re.search(r"^[#＃\-*]", text, re.M):
        return "meta_text"
    if re.search(r"資料\s*[0-9０-９]", text):
        return "numbered_doc_ref"
    if "〜" in text and "〜" not in question:
        return "placeholder_left"     # 書き出しの型の「〜」が置き換えられずに残っている
    if leaks_gold(text, gold, question):
        return "gold_leak"
    return None


def answer_quotes(cot_text: str) -> list[str]:
    """CoT 回答の逐語引用（正規化済み・短いものは除く）。"""
    return formatting.answer_quotes(cot_text)


# =============================================================================
# 差し替え候補
# =============================================================================

class NearPool:
    """コーパス全体から、質問に字面が近い差し替え候補を引く（文字 bigram の Jaccard）。"""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.norm = [formatting.norm_text(c.text) for c in chunks]
        self.bg = [{n[i:i + 2] for i in range(len(n) - 1)} for n in self.norm]

    def candidates(self, question: str, exclude_ids: set[str], exclude_texts: set[str],
                   quotes: list[str], k: int, ref_len: int) -> list[Chunk]:
        """質問に近い順に k 件。次のものは除く:
          - exclude_ids（その例に既にある文書・正解の文書）
          - exclude_texts と正規化後の本文が同じ文書（ID 違いの重複）
          - 回答の引用を含む文書（正解の内容を持っている）
        長さが ref_len の ±LEN_WINDOW に入るものを優先する（差し替えた例だけ文書が短い/長い、
        という形の手掛かりを残さない）。
        """
        qb = bigrams(question)
        if not qb:
            return []
        scored = []
        for c, n, b in zip(self.chunks, self.norm, self.bg):
            if c.id in exclude_ids or n in exclude_texts or any(q in n for q in quotes):
                continue
            inter = len(qb & b)
            if inter == 0:
                continue
            scored.append((inter / len(qb | b), c.id, c))
        scored.sort(key=lambda t: (-t[0], t[1]))
        lo, hi = (1 - LEN_WINDOW) * ref_len, (1 + LEN_WINDOW) * ref_len
        near = [c for _, _, c in scored if lo <= len(c.text) <= hi]
        far = [c for _, _, c in scored if not (lo <= len(c.text) <= hi)]
        return (near + far)[:k]


# =============================================================================
# 計画（決定的・LLM 非依存）
# =============================================================================

@dataclass
class RefusalTask:
    example_id: str
    question_id: str
    kind: str                 # absent / twin
    chunk_id: str             # 正解の文書（差し替える位置の印）
    question: str
    gold: str
    cot_text: str
    layout: list[str]         # 長さ調整後の文書 ID 列（正解の文書を含む）
    slot: int
    rank: int = 0             # 双子の採用順（シードで決まる）
    layout_key: str = field(default="")

    def __post_init__(self):
        if not self.layout_key:
            self.layout_key = format(stable_seed(self.kind, self.slot, *self.layout), "x")


@dataclass
class Plan:
    absent: list[RefusalTask]
    twin_buckets: dict[int, list[RefusalTask]]   # 文書数 → 採用順に並べた候補
    twin_quotas: dict[int, int]                  # 文書数 → 作る数
    n_answers: int = 0
    skipped_oversize: int = 0


def plan_tasks(
    assembled: list[AssembledRecord],
    answers_by_qid: dict[str, AnswerRecord],
    index: dict[str, Chunk],
    counter: Callable[[list[dict]], int],
    twin_ratio: float = 0.0,
    seed: int = 42,
    **fit_kwargs,
) -> Plan:
    """拒否例を作る対象と、その文書の並びを決める。

    並びは書き出しと同じ fit_record で長さ調整した後のもの。拒否文は「見せた資料」を前提に
    書かれるので、作文の前に並びを確定させる（absent は元の CoT 回答の長さで調整する。
    拒否文は CoT より短いのが普通で、書き出し時にも長さを確認する）。

    twin_ratio は「回答例のうち、双子（同じ質問の拒否版）も作る割合」。文書数ごとに同じ割合にする。
    """
    max_len = fit_kwargs.get("max_len", 2048)
    plan = Plan(absent=[], twin_buckets={}, twin_quotas={})
    answer_by_n: dict[int, list[RefusalTask]] = {}
    count_by_n: dict[int, int] = {}

    for rec in assembled:
        _, n_tok, _, order, _ = export.fit_record(rec, index, counter, seed=seed, **fit_kwargs)
        if n_tok > max_len:
            plan.skipped_oversize += 1       # 回答例としても捨てられる
            continue
        natural_miss = rec.refuse and rec.chunk_id not in order   # 検索が正解の文書を取り逃した例
        if not natural_miss and (not (rec.refuse or rec.oracle_included) or rec.chunk_id not in order):
            continue                          # 既定モードの oracle なし例は対象外
        ans = answers_by_qid.get(rec.example_id)
        gold = (ans.final_answer if ans and ans.final_answer.strip()
                else final_answer_of(rec.cot_text))
        base = dict(question_id=rec.example_id, chunk_id=rec.chunk_id, question=rec.question,
                    gold=gold, cot_text=rec.cot_text, layout=order,
                    slot=-1 if natural_miss else order.index(rec.chunk_id))
        if rec.refuse:
            plan.absent.append(RefusalTask(example_id=rec.example_id, kind="absent", **base))
        else:
            plan.n_answers += 1
            count_by_n[len(order)] = count_by_n.get(len(order), 0) + 1
            if twin_ratio > 0:
                answer_by_n.setdefault(len(order), []).append(
                    RefusalTask(example_id=make_twin_id(rec.example_id), kind="twin", **base))

    for n, tasks in sorted(answer_by_n.items()):
        tasks.sort(key=lambda t: stable_seed(seed, t.question_id, "twin"))
        for r, t in enumerate(tasks):
            t.rank = r
        plan.twin_buckets[n] = tasks
        plan.twin_quotas[n] = min(len(tasks), round(twin_ratio * count_by_n[n]))
    return plan


# =============================================================================
# LLM 呼び出し（自由文。構造化出力は使わない）
# =============================================================================

_client = None


def _get_client():
    global _client
    if _client is None:
        from llm_factory import create_openai_client
        _client = create_openai_client(async_mode=True)
        if _client is None:
            raise RuntimeError("LLM クライアントを作れません（接続情報を確認してください）")
    return _client


async def _chat(prompt: str, temperature: float, max_tokens: int) -> str:
    from llm_factory import get_model_name
    resp = await _get_client().chat.completions.create(
        model=get_model_name(),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature, max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def parse_verdict(text: str) -> Optional[bool]:
    """「判定: yes/no」を読む。指示の復唱に引きずられないよう最後に現れたものを採る。読めなければ None。"""
    ms = re.findall(r"判定[:：]\s*(yes|no)", text, re.I)
    if ms:
        return ms[-1].lower() == "yes"
    toks = re.findall(r"\b(yes|no)\b", text, re.I)
    return toks[-1].lower() == "yes" if toks else None


async def judge_grounded(question: str, gold: str, docs: list[Chunk]) -> bool:
    """文書群から模範解答が導けるか。判定が読めない場合は安全側（導ける = 拒否例に使わない）。"""
    prompt = get_user_prompt("raft_grounded", context=formatting.render_doc_list(docs),
                             question=question, answer=gold)
    verdict = parse_verdict(await _chat(prompt, temperature=0.0, max_tokens=200))
    return True if verdict is None else verdict


async def write_refusal(question: str, docs: list[Chunk], opener: str) -> str:
    prompt = get_user_prompt("raft_refusal", question=question, opener=opener,
                             context=formatting.render_doc_list(docs))
    return await _chat(prompt, temperature=0.8, max_tokens=700)


def mock_refusal(question: str, opener: str) -> str:
    """dry-run 用の決定的モック拒否文（検証を通る形）。"""
    topic = question.strip().rstrip("。？?")[:60]
    head = opener.replace("〜", topic)
    return (
        f"{head}\n\n"
        "提示された参考資料は、関連する周辺の事項や別の対象についての説明を扱っています。"
        "しかし、質問で問われている内容そのものに答える記述は、いずれの資料にも見当たりません。"
        "近い語句は現れるものの、問われている点を定めた箇所ではありません。\n\n"
        "資料に根拠が無いため、推測や一般的な知識で内容を補うことは控えます。"
        "根拠となる記述が確認できた場合に限り、該当する箇所を示したうえで回答します。\n\n"
        f"{FINAL_PREFIX}: 提供された参考資料には、この質問の答えとなる記載がありません。"
    )


# =============================================================================
# 1 例の処理
# =============================================================================

def _record(task: RefusalTask, **kw) -> RefusalRecord:
    return RefusalRecord(
        example_id=task.example_id, question_id=task.question_id, kind=task.kind,
        chunk_id=task.chunk_id, question=task.question, gold=task.gold,
        layout_key=task.layout_key, n_docs=len(task.layout), slot=task.slot, **kw,
    )


def ranked_candidates(task: RefusalTask, ranked: list[str], index: dict[str, Chunk],
                      layout_docs: list[Chunk], k: int) -> list[Chunk]:
    """検索の順位から差し替え候補を引く: その例にまだ無い文書を、順位の高い順に k 件。

    正解の文書・本文が同じ文書・回答の引用を含む文書は除く（NearPool.candidates と同じ決まり）。
    検索で次点だった文書なので、質問に近いが答えは載っていない文書になりやすい。
    """
    have = set(task.layout) | {task.chunk_id}
    texts = {formatting.norm_text(d.text) for d in layout_docs}
    quotes = answer_quotes(task.cot_text)
    out: list[Chunk] = []
    for cid in ranked:
        c = index.get(cid)
        if c is None or cid in have:
            continue
        n = formatting.norm_text(c.text)
        if n in texts or any(q in n for q in quotes):
            continue
        out.append(c)
        if len(out) >= k:
            break
    return out


async def make_refusal(task: RefusalTask, pool: NearPool, index: dict[str, Chunk],
                       dry_run: bool = False, ranked: Optional[list[str]] = None) -> RefusalRecord:
    """ranked（その質問の検索結果）を渡すと、差し替え候補を検索の順位から引く。無ければ字面の近さ。"""
    layout_docs = [index[i] for i in task.layout]
    if task.slot < 0:
        # 検索が正解の文書を取り逃した例。差し替えは無く、見せる文書群はそのまま。
        # その文書群から模範解答が導けるなら拒否例にできない（正解の文書以外でも答えられる質問）
        try:
            grounded = False if dry_run else await judge_grounded(task.question, task.gold, layout_docs)
        except Exception as e:
            return _record(task, status="error", reason=f"grounded_call: {type(e).__name__}: {e}"[:200])
        if grounded:
            return _record(task, status="skip", reason="answerable_from_retrieved")
        docs = layout_docs
        placed = dict(doc_order=[d.id for d in docs], substitute_id="", cand_rank=-1)
    else:
        golden = index.get(task.chunk_id)
        if golden is None:
            return _record(task, status="skip", reason="golden_not_in_corpus")
        cands = ranked_candidates(task, ranked, index, layout_docs, MAX_CANDIDATES) if ranked else []
        if not cands:
            cands = pool.candidates(
                task.question,
                exclude_ids=set(task.layout),
                exclude_texts={formatting.norm_text(d.text) for d in layout_docs},
                quotes=answer_quotes(task.cot_text),
                k=MAX_CANDIDATES, ref_len=len(golden.text),
            )
        if not cands:
            return _record(task, status="skip", reason="no_candidate")

        # 差し替え後の文書群で模範解答が導けないことを確かめる。導けたら次の候補へ
        docs, rank = None, -1
        for ci, c in enumerate(cands):
            trial = layout_docs[:task.slot] + [c] + layout_docs[task.slot + 1:]
            try:
                grounded = False if dry_run else await judge_grounded(task.question, task.gold, trial)
            except Exception as e:
                return _record(task, status="error", reason=f"grounded_call: {type(e).__name__}: {e}"[:200])
            if not grounded:
                docs, rank = trial, ci
                break
        if docs is None:
            return _record(task, status="skip", reason="all_candidates_grounded")
        placed = dict(doc_order=[d.id for d in docs], substitute_id=docs[task.slot].id, cand_rank=rank)

    opener_i = random.Random(stable_seed(task.example_id, "opener")).randrange(len(REFUSAL_OPENERS))
    opener = REFUSAL_OPENERS[opener_i]
    fails: list[str] = []
    for _ in range(MAX_WRITE_TRIES):
        try:
            text = (mock_refusal(task.question, opener) if dry_run
                    else await write_refusal(task.question, docs, opener))
        except Exception as e:
            return _record(task, status="error", fails=fails,
                           reason=f"write_call: {type(e).__name__}: {e}"[:200], **placed)
        why = validate_refusal(text, task.gold, task.question)
        if why is None:
            return _record(task, status="ok", refusal=text, fails=fails, opener=opener_i, **placed)
        fails.append(why)
    return _record(task, status="skip", reason="refusal_invalid", fails=fails, **placed)


# =============================================================================
# フェーズ実行
# =============================================================================

def _reusable(task: RefusalTask, done: dict[str, RefusalRecord]) -> bool:
    """checkpoint の記録を使い回せるか（error は再試行。並びが変わっていたら作り直す）。"""
    r = done.get(task.example_id)
    return bool(r) and r.status in ("ok", "skip") and r.layout_key == task.layout_key


def select_refusals(plan: Plan, records: dict[str, RefusalRecord]) -> list[RefusalRecord]:
    """書き出す拒否例を決定的に選ぶ: absent は ok の全件、twin は文書数ごとに採用順で枠まで。"""
    out = [records[t.example_id] for t in plan.absent
           if _reusable(t, records) and records[t.example_id].status == "ok"]
    for n, tasks in sorted(plan.twin_buckets.items()):
        ok = [records[t.example_id] for t in tasks
              if _reusable(t, records) and records[t.example_id].status == "ok"]
        out += ok[:plan.twin_quotas[n]]
    return out


def has_pending(plan: Plan, done: dict[str, RefusalRecord]) -> bool:
    """まだ LLM を呼ぶ必要がある例が残っているか（全て checkpoint で足りるなら False）。"""
    if any(not _reusable(t, done) for t in plan.absent):
        return True
    for n, tasks in plan.twin_buckets.items():
        n_ok = sum(1 for t in tasks if _reusable(t, done) and done[t.example_id].status == "ok")
        if n_ok < plan.twin_quotas[n] and any(not _reusable(t, done) for t in tasks):
            return True
    return False


async def run_refusal_phase(
    plan: Plan,
    pool: NearPool,
    index: dict[str, Chunk],
    concurrency: int,
    dry_run: bool = False,
    done: Optional[dict[str, RefusalRecord]] = None,
    sink=None,
    ranked_by_qid: Optional[dict[str, list[str]]] = None,
) -> dict[str, RefusalRecord]:
    """計画に沿って拒否例を作り、example_id → RefusalRecord を返す（done の使い回し分を含む）。

    ranked_by_qid（question_id → 検索の順位）を渡すと、双子の差し替え候補を検索の次点から引く。

    sink(dict) が与えられれば 1 例できるごとに追記（逐次チェックポイント）。
    双子は文書数ごとの枠が埋まるまで、検証落ちを見込んで少し多めに候補を出す。
    """
    records: dict[str, RefusalRecord] = dict(done or {})

    def emit(task: RefusalTask, rec):
        if rec is None:   # 予期しない例外（run_incremental が握りつぶしたもの）
            rec = _record(task, status="error", reason="unexpected_exception")
        records[task.example_id] = rec
        if sink:
            sink(rec.model_dump())

    async def run_batch(tasks: list[RefusalTask]):
        await _genutil.run_incremental(
            tasks, lambda t: make_refusal(t, pool, index, dry_run,
                                          ranked=(ranked_by_qid or {}).get(t.question_id)),
            concurrency, emit)

    await run_batch([t for t in plan.absent if not _reusable(t, records)])

    queues = {n: [t for t in tasks if not _reusable(t, records)]
              for n, tasks in plan.twin_buckets.items()}
    while True:
        batch: list[RefusalTask] = []
        for n, tasks in plan.twin_buckets.items():
            n_ok = sum(1 for t in tasks if _reusable(t, records)
                       and records[t.example_id].status == "ok")
            need = plan.twin_quotas[n] - n_ok
            if need <= 0 or not queues[n]:
                continue
            take = need + max(1, need // 8)
            batch += queues[n][:take]
            queues[n] = queues[n][take:]
        if not batch:
            break
        await run_batch(batch)
    return records
