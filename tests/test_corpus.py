import pytest
import corpus


@pytest.mark.asyncio
async def test_store_and_list(tmp_db):
    doc_id = await corpus.store_doc(tmp_db, "Doc A", "a.txt", "hello world")
    assert doc_id == 1
    docs = await corpus.list_docs(tmp_db)
    assert len(docs) == 1
    assert docs[0]["title"] == "Doc A"
    assert docs[0]["active"] is True


@pytest.mark.asyncio
async def test_duplicate_content_ignored(tmp_db):
    await corpus.store_doc(tmp_db, "Doc A", "a.txt", "hello world", label="v1")
    await corpus.store_doc(tmp_db, "Doc A2", "a2.txt", "hello world", label="v2")
    docs = await corpus.list_docs(tmp_db)
    assert len(docs) == 1, "duplicate content must not create a second row"
    assert docs[0]["label"] == "v2", "metadata should be updated on re-store"


@pytest.mark.asyncio
async def test_delete_doc(tmp_db):
    doc_id = await corpus.store_doc(tmp_db, "Doc", "d.txt", "text")
    await corpus.delete_doc(tmp_db, doc_id)
    docs = await corpus.list_docs(tmp_db)
    assert len(docs) == 0


@pytest.mark.asyncio
async def test_get_docs_by_ids(tmp_db):
    id1 = await corpus.store_doc(tmp_db, "A", "a.txt", "content a")
    id2 = await corpus.store_doc(tmp_db, "B", "b.txt", "content b")
    docs = await corpus.get_docs_by_ids(tmp_db, [id1])
    assert len(docs) == 1
    assert docs[0]["title"] == "A"
    assert "content a" in docs[0]["content"]


@pytest.mark.asyncio
async def test_get_docs_by_ids_empty(tmp_db):
    docs = await corpus.get_docs_by_ids(tmp_db, [])
    assert docs == []


@pytest.mark.asyncio
async def test_set_inactive_excluded_from_get(tmp_db):
    doc_id = await corpus.store_doc(tmp_db, "Doc", "d.txt", "text")
    await corpus.set_doc_active(tmp_db, doc_id, False)
    docs = await corpus.get_docs_by_ids(tmp_db, [doc_id])
    assert len(docs) == 0
