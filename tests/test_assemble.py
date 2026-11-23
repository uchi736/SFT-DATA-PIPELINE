from raft.schemas import Chunk, AnswerRecord
from raft import corpus as corpus_mod
from raft import assemble


def _chunks(n=6):
    return [Chunk(id=f"c{i}", text=f"本文{i}。" * 20) for i in range(n)]


def _answer(qid="c0__q0", chunk_id="c0"):
    return AnswerRecord(
        question_id=qid, chunk_id=chunk_id, question="質問?",
        golden_text="ゴールデン本文。", reasoning="r", final_answer="a",
        cot_text="推論。\n\n最終的な回答: a", ok=True,
    )


def test_assemble_deterministic_same_seed():
    chunks = _chunks()
    s1 = corpus_mod.make_distractor_sampler(chunks, seed=42)
    s2 = corpus_mod.make_distractor_sampler(chunks, seed=42)
    a1 = assemble.assemble_one(_answer(), s1, k=3, oracle_prob=0.8, seed=42)
    a2 = assemble.assemble_one(_answer(), s2, k=3, oracle_prob=0.8, seed=42)
    assert a1.doc_order == a2.doc_order
    assert a1.oracle_included == a2.oracle_included
    assert a1.distractor_ids == a2.distractor_ids


def test_distractors_exclude_golden_and_count():
    chunks = _chunks()
    sampler = corpus_mod.make_distractor_sampler(chunks, seed=7)
    a = assemble.assemble_one(_answer(), sampler, k=3, oracle_prob=1.0, seed=7)
    assert "c0" not in a.distractor_ids
    assert len(a.distractor_ids) == 3
    assert a.oracle_included is True
    # oracle 投入時は doc_order にゴールデンが含まれる
    assert "c0" in a.doc_order
    assert len(a.doc_order) == 4


def test_oracle_frequency_close_to_prob():
    chunks = _chunks()
    sampler = corpus_mod.make_distractor_sampler(chunks, seed=1)
    answers = [_answer(qid=f"c0__q{i}") for i in range(400)]
    assembled = assemble.run_phase3(answers, sampler, k=3, oracle_prob=0.8, seed=1)
    ratio = sum(1 for a in assembled if a.oracle_included) / len(assembled)
    assert 0.7 <= ratio <= 0.9


def test_oracle_forced_when_pool_empty():
    # コーパスがゴールデン1件のみ → ディストラクタ0 → oracle 強制
    chunks = [Chunk(id="only", text="本文。" * 20)]
    sampler = corpus_mod.make_distractor_sampler(chunks, seed=3)
    ans = _answer(qid="only__q0", chunk_id="only")
    a = assemble.assemble_one(ans, sampler, k=3, oracle_prob=0.0, seed=3)
    assert a.oracle_included is True
    assert a.distractor_ids == []
    assert a.doc_order == ["only"]


# ---- --distractor-mode retrieved ----

QO, QC = "##begin_quote##", "##end_quote##"


def _rchunks():
    texts = {
        "g": "安全弁は一年以内ごとに一回、定期に分解して点検しなければならない。" * 3,
        "dup": "別の規則でも、安全弁は一年以内ごとに一回、定期に分解して点検しなければならない。と定める。",
        "h1": "圧力計は、六月以内ごとに一回、比較検査を行わなければならない。" * 3,
        "h2": "水面計は、毎日一回以上、機能を点検しなければならない。" * 3,
        "h3": "給水装置は、月に一回、作動の状況を確認する。" * 4,
        "h4": "吹出し装置は、使用の前に漏れの有無を確認する。" * 4,
        "h5": "燃焼装置の火炎検出器は、汚れを取り除いて機能を確認する。" * 3,
        "r1": "技術訓練所では、溶接の技量を項目ごとに評価している。" * 3,
        "r2": "研究所では、若手が全国の工場に出掛けて情報を交換する。" * 3,
    }
    return [Chunk(id=k, text=v) for k, v in texts.items()]


def _ranswer():
    gold = _rchunks()[0].text
    quote = "安全弁は一年以内ごとに一回、定期に分解して点検しなければならない。"
    return AnswerRecord(
        question_id="g__q0", chunk_id="g", question="安全弁の分解点検の頻度は?",
        golden_text=gold, reasoning="r", final_answer="一年以内ごとに一回",
        cot_text=f"{QO} {quote} {QC} とある。\n\n最終的な回答: 一年以内ごとに一回", ok=True,
    )


def _retrieved(ranked, **kw):
    chunks = _rchunks()
    index = corpus_mod.index_corpus(chunks)
    sampler = corpus_mod.make_distractor_sampler(chunks, seed=5)
    kw.setdefault("k", 3)
    kw.setdefault("seed", 5)
    kw.setdefault("doc_order", "rank")
    return assemble.assemble_retrieved(_ranswer(), ranked, index, sampler, **kw)


def test_retrieved_shuffle_is_default_and_deterministic():
    chunks = _rchunks()
    index = corpus_mod.index_corpus(chunks)
    sampler = corpus_mod.make_distractor_sampler(chunks, seed=5)
    ranked = ["g", "h1", "h2", "h3", "h4"]
    orders = set()
    for i in range(40):
        ans = _ranswer().model_copy(update={"question_id": f"g__q{i}"})
        a = assemble.assemble_retrieved(ans, ranked, index, sampler, k=3, seed=5)
        b = assemble.assemble_retrieved(ans, ranked, index, sampler, k=3, seed=5)
        assert a.doc_order == b.doc_order and sorted(a.doc_order) == ["g", "h1", "h2", "h3"]
        orders.add(a.doc_order.index("g"))
    assert len(orders) == 4          # 正解の文書がどの位置にも現れる


def test_retrieved_hit_keeps_rank_order_and_golden():
    a = _retrieved(["h1", "g", "h2", "h3", "h4"])
    assert a.oracle_included is True and a.refuse is False
    assert a.doc_order == ["h1", "g", "h2", "h3"]          # 上位 k+1 件を検索の順位のまま
    assert a.distractor_ids == ["h1", "h2", "h3"]
    assert a.golden_rank == 1 and a.distractor_mode == "retrieved"
    assert a.user_content.index("圧力計") < a.user_content.index("安全弁は一年以内")


def test_retrieved_miss_becomes_absent_without_golden():
    ranked = ["h1", "h2", "h3", "h4", "h5", "g"]            # 正解は 6 位 = 文脈 4 件に入らない
    a = _retrieved(ranked)
    assert a.oracle_included is False and a.refuse is False
    assert "g" not in a.doc_order and len(a.doc_order) == 4
    assert a.assistant_content == _ranswer().cot_text       # absent-mode answer では回答のまま
    r = _retrieved(ranked, absent_mode="refuse")
    assert r.refuse is True and "g" not in r.doc_order      # 差し替える位置の印は置かない


def test_retrieved_drops_docs_that_carry_the_answer():
    # dup は回答の引用をそのまま含む → ディストラクタにせず、次の順位を繰り上げる
    a = _retrieved(["dup", "g", "h1", "h2", "h3"])
    assert "dup" not in a.doc_order and a.dropped_ids == ["dup"]
    assert a.doc_order == ["g", "h1", "h2", "h3"] and a.golden_rank == 0


def test_retrieved_fills_from_random_when_ranking_is_short():
    a = _retrieved(["g", "h1"])
    assert len(a.doc_order) == 4 and a.doc_order[:2] == ["g", "h1"]
    assert "dup" not in a.doc_order                         # 埋め合わせにも答えを含む文書は使わない


def test_retrieved_mix_random_replaces_only_distractors():
    ranked = ["g", "h1", "h2", "h3", "h4"]
    full = _retrieved(ranked, mix_random=1.0)
    assert full.doc_order[0] == "g" and full.n_mixed == 3
    assert not set(full.doc_order[1:]) & {"h1", "h2", "h3"}
    assert "dup" not in full.doc_order
    none = _retrieved(ranked, mix_random=0.0)
    assert none.n_mixed == 0 and none.doc_order == ["g", "h1", "h2", "h3"]
    assert _retrieved(ranked, mix_random=0.5).doc_order == _retrieved(ranked, mix_random=0.5).doc_order


def test_run_phase3_retrieved_requires_ranking():
    chunks = _rchunks()
    sampler = corpus_mod.make_distractor_sampler(chunks, seed=5)
    index = corpus_mod.index_corpus(chunks)
    out = assemble.run_phase3([_ranswer()], sampler, 3, 0.8, 5, distractor_mode="retrieved",
                              ranked_by_qid={"g__q0": ["h1", "g", "h2", "h3"]}, index=index,
                              doc_order="rank")
    assert out[0].doc_order == ["h1", "g", "h2", "h3"]
    try:
        assemble.run_phase3([_ranswer()], sampler, 3, 0.8, 5, distractor_mode="retrieved",
                            ranked_by_qid={}, index=index)
    except ValueError:
        pass
    else:
        raise AssertionError("検索結果が無い質問は誤りにする")
