import myfaiss.create_index as ci


class _FakeEnc:
    def encode(self, text: str):
        return list(range(len(text.split())))

    def decode(self, token_ids):
        return " ".join([f"tok{t}" for t in token_ids])


def test_load_docs_has_50_files():
    docs = ci.load_docs()
    assert len(docs) == 50
    assert all(d["doc_id"].endswith(".txt") for d in docs)


def test_chunk_by_token_overlap_and_non_empty(monkeypatch):
    monkeypatch.setattr(ci, "get_encoder", lambda: _FakeEnc())
    text = " ".join(["hello"] * 300)
    chunks = ci.chunk_by_token(text, chunk_size=40, overlap=10)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_build_chunk_records_contains_metadata(monkeypatch):
    monkeypatch.setattr(ci, "get_encoder", lambda: _FakeEnc())
    docs = [{"doc_id": "d1.txt", "text": "a " * 200}]
    chunks = ci.build_chunk_records(docs)
    assert chunks
    assert chunks[0]["doc_id"] == "d1.txt"
    assert chunks[0]["chunk_id"] == 0
    assert "text" in chunks[0]
