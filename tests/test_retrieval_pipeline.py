import app.retrieval.build_index as bi


class _FakeEnc:
    def encode(self, text: str):
        return list(range(len(text.split())))

    def decode(self, token_ids):
        return " ".join([f"tok{t}" for t in token_ids])


def test_load_docs_reads_txt_files():
    docs = bi.load_docs()
    assert docs
    assert all(d["doc_id"].endswith(".txt") for d in docs)


def test_chunk_by_token_overlap_and_non_empty(monkeypatch):
    monkeypatch.setattr(bi, "get_encoder", lambda: _FakeEnc())
    text = " ".join(["hello"] * 300)
    chunks = bi.chunk_by_token(text, chunk_size=40, overlap=10)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_build_chunk_records_contains_metadata(monkeypatch):
    monkeypatch.setattr(bi, "get_encoder", lambda: _FakeEnc())
    docs = [{"doc_id": "d1.txt", "text": "a " * 200}]
    chunks = bi.build_chunk_records(docs)
    assert chunks
    assert chunks[0]["doc_id"] == "d1.txt"
    assert chunks[0]["chunk_id"] == 0
    assert "text" in chunks[0]
