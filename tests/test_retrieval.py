from raft.schemas import AnswerRecord, Chunk, RetrievalRecord
from raft import retrieval


def _chunks():
    texts = [
        "ガス軸受は、潤滑剤として空気などの気体を用いる軸受である。",
        "転がり軸受は、内輪と外輪の間に玉やころなどの転動体を配置する。",
        "潤滑油の粘度は温度に強く依存し、温度が上がると粘度は低下する。",
        "メカニカルシールは、回転軸と静止部の間の流体漏れを防ぐ密封装置である。",
        "ボイラの安全弁は、圧力が設定値を超えたときに蒸気を逃がす。",
    ]
    return [Chunk(id=f"c{i}", text=t) for i, t in enumerate(texts)]


def _answer(qid, chunk_id, question):
    return AnswerRecord(question_id=qid, chunk_id=chunk_id, question=question,
                        golden_text="g", cot_text="推論。\n\n最終的な回答: a", ok=True)


def test_bm25_ranks_matching_doc_first():
    bm = retrieval.BM25([c.text for c in _chunks()])
    assert bm.ranked("潤滑油の粘度と温度の関係", 3)[0] == 2
    assert bm.ranked("安全弁はいつ蒸気を逃がすか", 3)[0] == 4


def test_rrf_prefers_docs_in_both_lists():
    # 1 は両方の表にあり、0 と 2 は片方だけ
    assert retrieval.rrf([[0, 1], [1, 2]], top_k=3)[0] == 1
    assert len(retrieval.rrf([[0, 1], [1, 2]], top_k=2)) == 2


def test_dry_run_retriever_uses_bm25_without_network():
    r = retrieval.Retriever(_chunks(), mode="hybrid", use_rerank=True, top_k=3, dry_run=True)
    assert r.mode == "bm25" and r.use_rerank is False
    rec = r.record(_answer("c2__q0", "c2", "潤滑油の粘度は温度でどう変わるか"))
    assert rec.ranked[0] == "c2"
    assert rec.golden_rank == 0 and rec.golden_bm25_rank == 0 and rec.golden_dense_rank == -1
    assert rec.scores == []


def test_golden_rank_is_minus_one_when_not_in_candidates():
    r = retrieval.Retriever(_chunks(), top_k=1, dry_run=True)
    rec = r.record(_answer("c0__q0", "c0", "潤滑油の粘度は温度でどう変わるか"))
    assert rec.golden_rank == -1


def test_run_retrieval_reuses_only_matching_config():
    chunks = _chunks()
    r = retrieval.Retriever(chunks, top_k=3, dry_run=True)
    answers = [_answer("c2__q0", "c2", "潤滑油の粘度"), _answer("c4__q0", "c4", "安全弁と蒸気")]
    stale = {"c2__q0": RetrievalRecord(question_id="c2__q0", chunk_id="c2", ranked=["c9"], config_key="old")}
    written = []
    out = retrieval.run_retrieval(answers, r, done=stale, sink=written.append)
    assert out["c2__q0"].ranked[0] == "c2"          # 設定の違う記録は使わず引き直す
    assert len(written) == 2
    # 同じ設定の記録はそのまま使い、何も書かない
    written.clear()
    again = retrieval.run_retrieval(answers, r, done=out, sink=written.append)
    assert written == [] and again["c4__q0"].ranked == out["c4__q0"].ranked


def test_config_key_changes_with_corpus_and_top_k():
    chunks = _chunks()
    a = retrieval.Retriever(chunks, top_k=3, dry_run=True).config_key
    b = retrieval.Retriever(chunks, top_k=5, dry_run=True).config_key
    c = retrieval.Retriever(chunks[:-1], top_k=3, dry_run=True).config_key
    assert len({a, b, c}) == 3


def test_summarize_hit_rates():
    recs = {
        "a": RetrievalRecord(question_id="a", chunk_id="x", golden_rank=0, golden_bm25_rank=0),
        "b": RetrievalRecord(question_id="b", chunk_id="x", golden_rank=4, golden_dense_rank=2),
        "c": RetrievalRecord(question_id="c", chunk_id="x", golden_rank=-1),
        "d": RetrievalRecord(question_id="d", chunk_id="x", golden_rank=7),
    }
    s = retrieval.summarize(recs, n_ctx=5)
    assert s["hit@1"] == 0.25 and s["hit@5"] == 0.5 and s["golden_in_candidates"] == 0.75
