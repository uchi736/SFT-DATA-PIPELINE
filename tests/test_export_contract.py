from raft.schemas import Chunk, AssembledRecord
from raft import export, formatting
from raft.formatting import DOC_OPEN
from raft.tokenizer_utils import make_token_counter

COUNTER = make_token_counter(dry_run=True)


def _mk_record(golden, distractors, question="質問?", cot="推論。\n\n最終的な回答: a", oracle=True):
    docs = ([golden] + distractors) if oracle else list(distractors)
    order = [d.id for d in docs]
    user = formatting.format_context(docs, question)
    rec = AssembledRecord(
        example_id=f"{golden.id}__q0", chunk_id=golden.id, question=question,
        cot_text=cot, distractor_ids=[d.id for d in distractors], oracle_included=oracle,
        doc_order=order, user_content=user, assistant_content=cot,
    )
    index = {d.id: d for d in docs}
    return rec, index


def test_contract_basic():
    golden = Chunk(id="g", text="ゴールデン本文。")
    dists = [Chunk(id=f"d{i}", text=f"妨害{i}。") for i in range(3)]
    rec, index = _mk_record(golden, dists)
    records, stats = export.run_phase4([rec], index, COUNTER, max_len=10000)
    assert stats["kept"] == 1 and len(records) == 1
    msgs = records[0].messages
    roles = [m["role"] for m in msgs]
    assert set(roles) <= {"system", "user", "assistant"}
    assert roles.index("user") < roles.index("assistant")
    assert all(m["content"].strip() for m in msgs)
    assert COUNTER(msgs) <= 10000


def test_contract_with_system_prompt():
    golden = Chunk(id="g", text="本文。")
    rec, index = _mk_record(golden, [Chunk(id="d0", text="妨害。")])
    records, _ = export.run_phase4([rec], index, COUNTER, max_len=10000, system_prompt="あなたは助手です。")
    assert records[0].messages[0]["role"] == "system"


def test_oversize_reduce_shrinks_distractors():
    golden = Chunk(id="g", text="短いゴールデン。")
    dists = [Chunk(id=f"d{i}", text="あ" * 400) for i in range(4)]
    rec, index = _mk_record(golden, dists)
    records, stats = export.run_phase4(
        [rec], index, COUNTER, max_len=400, oversize_policy="reduce", min_distractors=1,
    )
    assert stats["kept"] == 1
    assert stats["reduced"] == 1
    # ゴールデン + 削減後の少数ディストラクタのみ
    assert records[0].messages[-2]["content"].count(DOC_OPEN) < 5
    # ゴールデンは残っている
    assert "短いゴールデン" in records[0].messages[-2]["content"]


def test_oversize_drop_policy():
    golden = Chunk(id="g", text="短いゴールデン。")
    dists = [Chunk(id=f"d{i}", text="あ" * 400) for i in range(4)]
    rec, index = _mk_record(golden, dists)
    records, stats = export.run_phase4(
        [rec], index, COUNTER, max_len=400, oversize_policy="drop",
    )
    assert stats["kept"] == 0
    assert stats["dropped_oversize"] == 1


def test_duplicate_pairs_removed():
    golden = Chunk(id="g", text="本文。")
    d = [Chunk(id="d0", text="妨害。")]
    rec1, idx1 = _mk_record(golden, d)
    rec2, _ = _mk_record(golden, d)  # 同一 user/assistant
    records, stats = export.run_phase4([rec1, rec2], idx1, COUNTER, max_len=10000)
    assert stats["kept"] == 1
    assert stats["dropped_dup"] == 1
