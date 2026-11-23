"""
質問の判定（LLM 非依存・決定的）: 生成した質問を 1 問ずつ 自動採用 / 要確認 / 自動除外 に分け、理由を付ける。

Phase1 の検証は「空でないこと」だけなので、質の悪い質問もそのまま学習例になる。ここでは
文字列の照合と検索の結果だけで分かる不備を拾う。意味の正しさ（質問が自然か、回答が正しいか）は
見ていない。そこは人か LLM の確認が要る。

判定と条件:
  reject（自動除外）… 学習例に使わない
    deictic            指示語で文書を指している（「この文書では」「上記の」など。質問だけでは何の話か分からない）
    answer_in_question 最終回答の中身が質問文にほぼ入っている（答えを読み取る練習にならない）
    duplicate          正規化した文面が、先に出た別の質問と同じ
  review（要確認）… 既定では学習例に使わず、確認用ファイルに回す
    weak_deictic       文脈に頼る語がある（「本研究」「同条」など。文書名が無いと特定できないことが多い）
    verbatim_copy      本文の長い写しを含む（実際の利用者はこう聞かない。検索が字面だけで当たる）
    near_duplicate     別の質問とほぼ同じ文面
    golden_not_found   正解の文書が検索の候補に入らない（質問が文書を特定できていない可能性）
    length             質問が短すぎる / 長すぎる
  accept（自動採用）… 上のどれにも当たらない
  注記（判定は変えず、理由として残すだけ）
    answer_elsewhere   検索の上位に、回答の引用をすべて含む別の文書がある（正解の文書が 1 つに決まらない）。
                       質問の不備ではなくコーパスの重なりなので除外しない。その文書は合成の段が文脈から外す。
                       正解と同じ文書から切り出した別のチャンクで起きやすい

閾値はこのファイルの定数。変えたら manifest に残る（question_check.thresholds）。
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))  # scripts/
from raft.schemas import AnswerRecord, QuestionReview, RetrievalRecord  # noqa: E402
from raft import formatting  # noqa: E402

VERDICTS = ("accept", "review", "reject")

THRESHOLDS = {
    # 最終回答の文字 3-gram のうち質問文に無いものの割合がこれ未満なら漏れ。
    # 質問の言い回しを繰り返したうえで数値だけを新しく答える正常な回答は、この割合が 0.1 を下回ることがある。
    # 閾値を上げすぎるとそうした回答まで除外される
    "answer_novel_min": 0.05,
    "answer_min_chars": 8,        # これより短い最終回答は漏れの判定をしない（3-gram が少なすぎる）
    "verbatim_min_chars": 30,     # 質問と本文の共通部分列がこの長さ以上なら写し
    "near_dup_jaccard": 0.85,     # 文字 2-gram の Jaccard がこれ以上ならほぼ同じ
    "near_dup_max_n": 4000,       # 質問がこれより多いときは near_duplicate を省く（総当たりのため）
    "min_chars": 12,
    "max_chars": 220,
}

LABELS = {
    "deictic": "指示語で文書を指している",
    "answer_in_question": "答えが質問文に入っている",
    "duplicate": "ほかの質問と同じ文面",
    "weak_deictic": "文脈に頼る語がある",
    "verbatim_copy": "本文の長い写しを含む",
    "near_duplicate": "ほかの質問とほぼ同じ文面",
    "answer_elsewhere": "正解の文書以外でも答えられる",
    "golden_not_found": "正解の文書が検索の候補に入らない",
    "length": "質問の長さが範囲外",
}
REJECT_CODES = ("deictic", "answer_in_question", "duplicate")
NOTE_CODES = ("answer_elsewhere",)      # 判定を変えない注記

# 質問だけを読んだ人には指す先が分からない表現
_DEICTIC = re.compile(
    r"(この|その|上記の?|前述の?|上述の?|下記の?|以下の|次の)(文書|文章|資料|記事|テキスト|本文|コンテキスト|記述|抜粋|引用)"
    r"|(本文書|本資料|本テキスト|当該文書|当該資料)"
    r"|(コンテキスト|与えられた|提示された|提供された)"
    r"|(文中|本文中|文書中|資料中|文書内|資料内)(で|に|の)"
    # 「文書に記載されている物質の中で」「文書の内容に基づいて」のような言い方
    r"|(文書|文章|資料|記事|テキスト|本文)(に|で|では|には|において|によると|によれば|から)は?、?"
    r"(記載|記述|書かれ|述べ|示され|挙げ|掲げ|説明|言及|定義|紹介|報告|よる|読み取)"
    r"|(文書|文章|資料|記事|テキスト|本文)の(内容|記載|記述|中)(に|で|から|を)"
)
# 文書名や法令名が質問に無いと特定できないことが多い表現
_WEAK_DEICTIC = re.compile(
    r"(本研究|本稿|本報|本論文|本報告|本開発|本システム|本装置|本手法|本試験|本条|本項|本号|本表|本図|本規程|本規則)"
    r"|(同条|同項|同号|同表|前条|前項|前号|次条|次項)"
    # 「その装置」のような、質問の中の語を受け直す言い方は当てない（誤検知になる）
    r"|この(研究|開発|試験|実験|条文|規定|規程|表|図|事例)"
    r"|(筆者|著者)"
)


def _norm_q(q: str) -> str:
    return formatting.norm_text(q).lower()


def _grams(s: str, n: int) -> set[str]:
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def answer_novelty(question: str, final_answer: str) -> Optional[float]:
    """最終回答の文字 3-gram のうち、質問文に現れないものの割合。判定できないときは None。"""
    a = formatting.norm_text(final_answer)
    if len(a) < THRESHOLDS["answer_min_chars"]:
        return None
    ag = _grams(a, 3)
    if not ag:
        return None
    return len(ag - _grams(formatting.norm_text(question), 3)) / len(ag)


def longest_common_substring(a: str, b: str, cap: int = 200) -> int:
    """a と b の最長共通部分列の長さ（文字）。cap に達したら打ち切る。"""
    if not a or not b:
        return 0
    best = 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b, 1):
            if ca == cb:
                v = prev[j - 1] + 1
                cur[j] = v
                if v > best:
                    best = v
                    if best >= cap:
                        return best
        prev = cur
    return best


def _reason(code: str, detail: str = "") -> dict[str, str]:
    return {"code": code, "label": LABELS[code], "detail": detail}


def check_one(ans: AnswerRecord, ret: Optional[RetrievalRecord] = None,
              carried_by: Optional[list[str]] = None) -> list[dict[str, str]]:
    """1 問だけで決まる条件を調べ、当たった理由の列を返す（重複の判定は check_all が足す）。

    carried_by: 検索の上位にあった、回答の引用をすべて含む別の文書の ID。
    """
    q = ans.question.strip()
    out: list[dict[str, str]] = []

    m = _DEICTIC.search(q)
    if m:
        out.append(_reason("deictic", m.group(0)))
    else:
        m = _WEAK_DEICTIC.search(q)
        if m:
            out.append(_reason("weak_deictic", m.group(0)))

    nov = answer_novelty(q, ans.final_answer)
    if nov is not None and nov < THRESHOLDS["answer_novel_min"]:
        out.append(_reason("answer_in_question", f"回答のうち質問文に無い部分 {nov:.2f}"))

    qn, gn = formatting.norm_text(q), formatting.norm_text(ans.golden_text)
    lcs = longest_common_substring(qn, gn, cap=THRESHOLDS["verbatim_min_chars"] + 40)
    if lcs >= THRESHOLDS["verbatim_min_chars"]:
        out.append(_reason("verbatim_copy", f"本文と一致する部分 {lcs} 文字"))

    if not (THRESHOLDS["min_chars"] <= len(q) <= THRESHOLDS["max_chars"]):
        out.append(_reason("length", f"{len(q)} 文字"))

    if carried_by:
        out.append(_reason("answer_elsewhere", "検索の上位に同じ内容の文書 " + ", ".join(carried_by[:3])))
    if ret is not None and ret.golden_rank < 0:
        out.append(_reason(
            "golden_not_found",
            f"BM25 {_rank_text(ret.golden_bm25_rank)} / 埋め込み {_rank_text(ret.golden_dense_rank)}"))
    return out


def _rank_text(r: int) -> str:
    return f"{r + 1} 位" if r >= 0 else "圏外"


def verdict_of(reasons: list[dict[str, str]]) -> str:
    if any(r["code"] in REJECT_CODES for r in reasons):
        return "reject"
    return "review" if any(r["code"] not in NOTE_CODES for r in reasons) else "accept"


def check_all(
    answers: list[AnswerRecord],
    retrieved: Optional[dict[str, RetrievalRecord]] = None,
    carried_by: Optional[dict[str, list[str]]] = None,
    overrides: Optional[dict[str, str]] = None,
) -> list[QuestionReview]:
    """成功した回答の質問をすべて判定する。戻り値は answers と同じ順。

    overrides: question_id → verdict（人の判定）。機械の判定を上書きし、元の判定を override に残す。
    """
    targets = [a for a in answers if a.ok and a.cot_text.strip()]
    reasons = {a.question_id: check_one(a, (retrieved or {}).get(a.question_id),
                                        (carried_by or {}).get(a.question_id))
               for a in targets}

    # 同じ文面（先に出た方を残す）
    first: dict[str, str] = {}
    norms = {a.question_id: _norm_q(a.question) for a in targets}
    for a in targets:
        n = norms[a.question_id]
        if n in first:
            reasons[a.question_id].append(_reason("duplicate", f"{first[n]} と同じ"))
        else:
            first[n] = a.question_id

    # ほぼ同じ文面（総当たり。件数が多いときは省く）
    if len(targets) <= THRESHOLDS["near_dup_max_n"]:
        bg = {a.question_id: _grams(norms[a.question_id], 2) for a in targets}
        ids = [a.question_id for a in targets]
        th = THRESHOLDS["near_dup_jaccard"]
        for i, qi in enumerate(ids):
            if any(r["code"] == "duplicate" for r in reasons[qi]) or not bg[qi]:
                continue
            for qj in ids[:i]:
                if not bg[qj] or norms[qi] == norms[qj]:
                    continue
                small, large = sorted((len(bg[qi]), len(bg[qj])))
                if small / large < th:        # 大きさの比だけで届かない組は飛ばす
                    continue
                inter = len(bg[qi] & bg[qj])
                if inter / (len(bg[qi]) + len(bg[qj]) - inter) >= th:
                    reasons[qi].append(_reason("near_duplicate", f"{qj} とほぼ同じ"))
                    break

    out: list[QuestionReview] = []
    for a in targets:
        rs = reasons[a.question_id]
        verdict, override = verdict_of(rs), ""
        ov = (overrides or {}).get(a.question_id)
        if ov in VERDICTS and ov != verdict:
            verdict, override = ov, verdict
        out.append(QuestionReview(
            question_id=a.question_id, chunk_id=a.chunk_id, question=a.question,
            final_answer=a.final_answer, verdict=verdict, reasons=rs, override=override))
    return out


def summarize(reviews: list[QuestionReview]) -> dict:
    """判定の件数と理由の内訳（manifest 用）。1 問に理由が複数あれば、それぞれに数える。"""
    return {
        "n": len(reviews),
        "verdicts": {v: sum(1 for r in reviews if r.verdict == v) for v in VERDICTS},
        "reasons": dict(sorted(Counter(x["code"] for r in reviews for x in r.reasons).items())),
        "overridden": sum(1 for r in reviews if r.override),
        "thresholds": dict(THRESHOLDS),
    }


def load_overrides(path) -> dict[str, str]:
    """人の判定を読む。JSONL で 1 行 {"question_id", "verdict"}（確認用ファイルを編集したものでよい）。"""
    import json
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            v = d.get("verdict")
            if d.get("question_id") and v in VERDICTS:
                out[d["question_id"]] = v
    return out


def usable_ids(reviews: list[QuestionReview], review_policy: str = "hold") -> set[str]:
    """学習例に使う質問の ID。review_policy: hold=要確認は使わない / keep=要確認も使う。"""
    ok = ("accept", "review") if review_policy == "keep" else ("accept",)
    return {r.question_id for r in reviews if r.verdict in ok}
