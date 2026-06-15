import json
from raft import corpus as corpus_mod


def _write(tmp_path, rows):
    p = tmp_path / "corpus.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(p)


def _rows_source_doc():
    # 実コーパス互換 {text, source, doc}（id/meta 無し）
    return [
        {"text": "あ" * 300, "source": "regulation", "doc": "01_anewho"},
        {"text": "い" * 300, "source": "regulation", "doc": "01_anewho"},
        {"text": "う" * 300, "source": "manual", "doc": "02_bar"},
        {"text": "短い", "source": "misc", "doc": "note"},  # 短い→ゴールデン候補外
    ]


def test_loads_source_doc_schema(tmp_path):
    path = _write(tmp_path, _rows_source_doc())
    chunks = corpus_mod.load_corpus(path)
    assert len(chunks) == 4
    # 全件にユニークID
    ids = [c.id for c in chunks]
    assert all(ids) and len(set(ids)) == 4
    # source/doc が meta へ畳み込まれている
    assert chunks[0].meta.get("source") == "regulation"
    assert chunks[0].meta.get("doc") == "01_anewho"


def test_ids_are_content_stable(tmp_path):
    path = _write(tmp_path, _rows_source_doc())
    ids1 = [c.id for c in corpus_mod.load_corpus(path)]
    ids2 = [c.id for c in corpus_mod.load_corpus(path)]
    assert ids1 == ids2  # 再ロードで同一（内容ハッシュ）


def test_ids_position_independent_under_limit(tmp_path):
    path = _write(tmp_path, _rows_source_doc())
    full = [c.id for c in corpus_mod.load_corpus(path)]
    limited = [c.id for c in corpus_mod.load_corpus(path, limit=2)]
    # --limit でも先頭2件のIDは不変（位置非依存）
    assert limited == full[:2]


def test_explicit_id_is_honored(tmp_path):
    path = _write(tmp_path, [{"text": "x" * 300, "id": "myid", "meta": {"k": 1}}])
    chunks = corpus_mod.load_corpus(path)
    assert chunks[0].id == "myid"
    assert chunks[0].meta == {"k": 1}


def test_split_text_respects_max_chars():
    text = "".join("あ。" for _ in range(300))  # 600字
    segs = corpus_mod.split_text(text, 200)
    assert len(segs) >= 2
    assert all(len(s) <= 200 for s in segs)


def test_split_text_short_is_noop():
    assert corpus_mod.split_text("短い文。", 200) == ["短い文。"]


def test_load_corpus_max_doc_chars(tmp_path):
    path = _write(tmp_path, [{"text": "".join("あ。" for _ in range(300)),
                              "source": "s", "doc": "d"}])
    chunks = corpus_mod.load_corpus(path, max_doc_chars=200)
    assert len(chunks) >= 2
    assert all(len(c.text) <= 200 for c in chunks)
    assert all(c.meta.get("doc") == "d" for c in chunks)   # provenance 保持
    assert all("part" in c.meta for c in chunks)            # 分割インデックス付与
    assert len({c.id for c in chunks}) == len(chunks)       # ID は一意


def test_golden_candidates_and_sampler(tmp_path):
    path = _write(tmp_path, _rows_source_doc())
    chunks = corpus_mod.load_corpus(path)
    goldens = corpus_mod.golden_candidates(chunks, min_chars=200)
    assert len(goldens) == 3  # 短いnoteは除外
    sampler = corpus_mod.make_distractor_sampler(chunks, seed=1)
    g = goldens[0]
    dists = sampler(g.id, k=2, salt="d")
    assert g.id not in [d.id for d in dists]
    assert len(dists) == 2
