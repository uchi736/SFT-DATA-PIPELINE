import asyncio

from raft import assemble, corpus as corpus_mod, refusal
from raft.tokenizer_utils import make_token_counter

from test_assemble import _rchunks, _ranswer


def _setup(ranked, twin_ratio=0.0, absent_mode="refuse"):
    chunks = _rchunks()
    index = corpus_mod.index_corpus(chunks)
    sampler = corpus_mod.make_distractor_sampler(chunks, seed=5)
    ans = _ranswer()
    asm = assemble.run_phase3([ans], sampler, 3, 0.8, 5, absent_mode=absent_mode,
                              distractor_mode="retrieved", ranked_by_qid={ans.question_id: ranked},
                              index=index, doc_order="rank")
    plan = refusal.plan_tasks(asm, {ans.question_id: ans}, index, make_token_counter(dry_run=True),
                              twin_ratio=twin_ratio, seed=5, max_len=100000)
    return chunks, index, ans, asm, plan


def test_natural_miss_is_planned_without_substitution():
    ranked = ["h1", "h2", "h3", "h4", "h5", "g"]
    chunks, index, ans, asm, plan = _setup(ranked)
    assert len(plan.absent) == 1
    t = plan.absent[0]
    assert t.slot == -1 and t.layout == ["h1", "h2", "h3", "h4"]
    rec = asyncio.run(refusal.make_refusal(t, refusal.NearPool(chunks), index, dry_run=True))
    assert rec.status == "ok" and rec.doc_order == t.layout
    assert rec.substitute_id == "" and rec.slot == -1
    assert refusal.is_refusal(rec.refusal)


def test_twin_substitute_comes_from_next_ranked_doc():
    ranked = ["h1", "g", "h2", "h3", "dup", "h4", "h5"]
    chunks, index, ans, asm, plan = _setup(ranked, twin_ratio=1.0)
    assert plan.absent == [] and plan.twin_quotas == {4: 1}
    t = plan.twin_buckets[4][0]
    assert t.slot == 1
    rec = asyncio.run(refusal.make_refusal(t, refusal.NearPool(chunks), index, dry_run=True,
                                           ranked=ranked))
    # 次点のうち dup は回答の引用を含むので飛ばし、h4 が入る。位置は正解の文書と同じ
    assert rec.status == "ok" and rec.substitute_id == "h4"
    assert rec.doc_order == ["h1", "h4", "h2", "h3"]


def test_ranked_candidates_skip_docs_already_shown():
    ranked = ["h1", "g", "h2", "h3", "h4", "h5"]
    chunks, index, ans, asm, plan = _setup(ranked, twin_ratio=1.0)
    t = plan.twin_buckets[4][0]
    cands = refusal.ranked_candidates(t, ranked, index, [index[i] for i in t.layout], k=3)
    assert [c.id for c in cands] == ["h4", "h5"]
