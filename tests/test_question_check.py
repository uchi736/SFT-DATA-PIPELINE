from raft.schemas import AnswerRecord, RetrievalRecord
from raft import question_check as qc

GOLD = ("ボイラの安全弁は、最高使用圧力を超えないように蒸気を逃がす装置である。"
        "安全弁は一年以内ごとに一回、定期に分解して点検しなければならない。")


def _ans(qid, question, final="一年以内ごとに一回です。", gold=GOLD):
    return AnswerRecord(question_id=qid, chunk_id=qid.split("__")[0], question=question,
                        golden_text=gold, final_answer=final,
                        cot_text=f"推論。\n\n最終的な回答: {final}", ok=True)


def _codes(review):
    return [r["code"] for r in review.reasons]


def test_plain_question_is_accepted():
    r = qc.check_all([_ans("c0__q0", "ボイラの安全弁は、どのくらいの頻度で分解点検が必要ですか。")])[0]
    assert r.verdict == "accept" and r.reasons == []


def test_deictic_is_rejected_with_matched_text():
    r = qc.check_all([_ans("c0__q0", "この文書では、安全弁の点検の頻度をどう定めていますか。")])[0]
    assert r.verdict == "reject" and _codes(r) == ["deictic"]
    assert r.reasons[0]["detail"] == "この文書" and r.reasons[0]["label"]


def test_deictic_variants_seen_in_first_dataset():
    for q in ("文書に記載されている物質の中で、別名が安全弁と呼ばれるものの点検の頻度は何ですか。",
              "安全弁の点検の頻度を、文書の内容に基づいて答えてください。",
              "文書内の安全弁について、点検の頻度はどのように記載されていますか。"):
        r = qc.check_all([_ans("c0__q0", q)])[0]
        assert r.verdict == "reject" and "deictic" in _codes(r), q
    # 文書の種類を名指しするだけの質問は当てない
    ok = qc.check_all([_ans("c0__q0", "技術資料の作成を担当する部署は、安全弁の点検の頻度をどう定めるか。")])[0]
    assert "deictic" not in _codes(ok)


def test_weak_deictic_goes_to_review_but_self_reference_does_not():
    weak = qc.check_all([_ans("c0__q0", "前項の規定により点検を行う頻度は、どのように定められていますか。")])[0]
    assert weak.verdict == "review" and _codes(weak) == ["weak_deictic"]
    # 「その装置」は質問の中の語を受け直しているだけなので当てない
    ok = qc.check_all([_ans("c0__q0", "蒸気を逃がす装置は何か、またその装置の点検の頻度はどれだけか。")])[0]
    assert "weak_deictic" not in _codes(ok)


def test_answer_in_question_only_when_nothing_new():
    leak = qc.check_all([_ans("c0__q0", "規格 XYZ K 9901(1999) に対応する別の鋼種の規格は何ですか。",
                              final="XYZ K 9901(1999)")])[0]
    assert leak.verdict == "reject" and "answer_in_question" in _codes(leak)
    # 質問の言い回しを繰り返しても、新しい値を答えていれば漏れではない
    fine = qc.check_all([_ans("c0__q0", "安全弁は何年以内ごとに分解して点検しなければならないか。",
                              final="一年以内ごとに分解して点検しなければならない。")])[0]
    assert "answer_in_question" not in _codes(fine)


def test_verbatim_copy_goes_to_review():
    q = "安全弁は一年以内ごとに一回、定期に分解して点検しなければならないとされる装置は何を逃がすか。"
    r = qc.check_all([_ans("c0__q0", q, final="蒸気です。蒸気を逃がします。")])[0]
    assert r.verdict == "review" and "verbatim_copy" in _codes(r)


def test_duplicate_keeps_first_and_near_duplicate_reviews():
    a = _ans("c0__q0", "ボイラの安全弁は、どのくらいの頻度で分解点検が必要ですか。")
    b = _ans("c1__q0", "ボイラの安全弁は、どのくらいの頻度で分解点検が必要ですか。 ")
    c = _ans("c2__q0", "ボイラの安全弁は、どのくらいの頻度で分解点検が必要ですか？")
    ra, rb, rc = qc.check_all([a, b, c])
    assert ra.verdict == "accept"
    assert rb.verdict == "reject" and "duplicate" in _codes(rb)
    assert rc.verdict == "review" and "near_duplicate" in _codes(rc)


def test_retrieval_based_reasons():
    a = _ans("c0__q0", "ボイラの安全弁は、どのくらいの頻度で分解点検が必要ですか。")
    ret = {"c0__q0": RetrievalRecord(question_id="c0__q0", chunk_id="c0", golden_rank=-1,
                                     golden_bm25_rank=12, golden_dense_rank=-1)}
    r = qc.check_all([a], retrieved=ret, carried_by={"c0__q0": ["c7"]})[0]
    assert r.verdict == "review"
    assert set(_codes(r)) == {"golden_not_found", "answer_elsewhere"}
    detail = next(x["detail"] for x in r.reasons if x["code"] == "golden_not_found")
    assert "13 位" in detail and "圏外" in detail
    # 同じ内容の文書がほかにあるだけなら、判定は変えず注記として残す
    note = qc.check_all([a], carried_by={"c0__q0": ["c7"]})[0]
    assert note.verdict == "accept" and _codes(note) == ["answer_elsewhere"]


def test_override_and_usable_ids():
    a = _ans("c0__q0", "前項の規定により点検を行う頻度は、どのように定められていますか。")
    b = _ans("c1__q0", "この文書では、安全弁の点検の頻度をどう定めていますか。")
    c = _ans("c2__q0", "安全弁の役割は何ですか。蒸気との関係を答えてください。", final="蒸気を逃がすことです。")
    plain = qc.check_all([a, b, c])
    assert qc.usable_ids(plain, "hold") == {"c2__q0"}
    assert qc.usable_ids(plain, "keep") == {"c0__q0", "c2__q0"}
    over = qc.check_all([a, b, c], overrides={"c0__q0": "accept"})
    assert over[0].verdict == "accept" and over[0].override == "review"
    assert qc.usable_ids(over, "hold") == {"c0__q0", "c2__q0"}
    s = qc.summarize(over)
    assert s["verdicts"] == {"accept": 2, "review": 0, "reject": 1} and s["overridden"] == 1


def test_failed_answers_are_not_reviewed():
    bad = AnswerRecord(question_id="c9__q0", chunk_id="c9", question="q", golden_text="g", ok=False)
    assert qc.check_all([bad]) == []
