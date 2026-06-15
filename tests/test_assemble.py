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
