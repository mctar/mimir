import numpy as np
import pytest
import corpus


@pytest.mark.asyncio
async def test_store_and_list(tmp_db):
    emb = np.ones(768, dtype=np.float32)
    doc_id = await corpus.store_doc(tmp_db, "Doc A", "a.txt", "hello world", emb)
    assert doc_id == 1
    docs = await corpus.list_docs(tmp_db)
    assert len(docs) == 1
    assert docs[0]["title"] == "Doc A"
    assert docs[0]["active"] is True


@pytest.mark.asyncio
async def test_duplicate_content_raises(tmp_db):
    emb = np.ones(768, dtype=np.float32)
    await corpus.store_doc(tmp_db, "Doc A", "a.txt", "hello world", emb)
    with pytest.raises(Exception):
        await corpus.store_doc(tmp_db, "Doc A2", "a2.txt", "hello world", emb)


@pytest.mark.asyncio
async def test_load_and_search(tmp_db):
    emb_a = np.array([1.0] + [0.0] * 767, dtype=np.float32)
    emb_b = np.array([0.0, 1.0] + [0.0] * 766, dtype=np.float32)
    await corpus.store_doc(tmp_db, "A", "a.txt", "content a", emb_a)
    await corpus.store_doc(tmp_db, "B", "b.txt", "content b", emb_b)

    docs = await corpus.load_corpus(tmp_db)
    assert len(docs) == 2

    query = np.array([1.0] + [0.0] * 767, dtype=np.float32)
    results = corpus.search_corpus(query, docs, k=1)
    assert results[0]["title"] == "A"
    assert results[0]["score"] > 0.99


@pytest.mark.asyncio
async def test_set_inactive_excluded_from_load(tmp_db):
    emb = np.ones(768, dtype=np.float32)
    doc_id = await corpus.store_doc(tmp_db, "Doc", "d.txt", "text", emb)
    await corpus.set_doc_active(tmp_db, doc_id, False)
    docs = await corpus.load_corpus(tmp_db)
    assert len(docs) == 0


@pytest.mark.asyncio
async def test_delete_doc(tmp_db):
    emb = np.ones(768, dtype=np.float32)
    doc_id = await corpus.store_doc(tmp_db, "Doc", "d.txt", "text", emb)
    await corpus.delete_doc(tmp_db, doc_id)
    docs = await corpus.list_docs(tmp_db)
    assert len(docs) == 0


def test_search_corpus_empty():
    results = corpus.search_corpus(np.ones(768, dtype=np.float32), [], k=5)
    assert results == []


def test_synthesis_prompt_no_corpus():
    prompt = corpus.build_synthesis_user_prompt("transcript text", [])
    assert "transcript text" in prompt
    assert "CORPUS" not in prompt


def test_synthesis_prompt_with_corpus():
    passages = [{"title": "Study A", "content": "key finding here"}]
    prompt = corpus.build_synthesis_user_prompt("transcript text", passages)
    assert "Study A" in prompt
    assert "CORPUS" in prompt
