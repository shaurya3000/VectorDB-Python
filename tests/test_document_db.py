
from document_db import DocumentDB


def test_document_insert_and_search(tmp_path):
    db = DocumentDB(tmp_path / "documents.db")

    doc_id = db.insert(
        "Python basics",
        "Python is a programming language.",
        [1.0, 0.0, 0.0],
    )

    assert db.count == 1
    assert db.list_documents()[0]["title"] == "Python basics"

    results = db.search([1.0, 0.0, 0.0], k=1)
    assert len(results) == 1
    assert results[0]["id"] == doc_id
    assert results[0]["distance"] < 0.001


def test_documents_persist_after_restart(tmp_path):
    db_path = tmp_path / "documents.db"

    db1 = DocumentDB(db_path)
    db1.insert(
        "Saved document",
        "This document should persist.",
        [1.0, 0.0],
    )

    db2 = DocumentDB(db_path)

    assert db2.count == 1
    assert db2.list_documents()[0]["title"] == "Saved document"


def test_document_delete(tmp_path):
    db = DocumentDB(tmp_path / "documents.db")
    doc_id = db.insert(
        "Temporary",
        "Delete this document.",
        [1.0, 0.0],
    )

    assert db.delete(doc_id) is True
    assert db.count == 0
    assert db.delete(doc_id) is False


def test_chunk_text():
    chunks = DocumentDB.chunk_text(
        "one two three four five six",
        chunk_words=4,
        overlap_words=1,
    )

    assert chunks == [
        "one two three four",
        "four five six",
    ]
