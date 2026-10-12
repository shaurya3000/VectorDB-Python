
from storage import VectorStorage


def test_vector_storage_save_and_load(tmp_path):
    db_file = tmp_path / "test_vectors.db"
    storage = VectorStorage(db_file)

    storage.save(
        1,
        "test-vector",
        "test",
        [1.0, 2.0, 3.0]
    )

    records = storage.load()

    assert len(records) == 1
    assert records[0]["id"] == 1
    assert records[0]["metadata"] == "test-vector"
    assert records[0]["category"] == "test"
    assert records[0]["embedding"] == [1.0, 2.0, 3.0]


def test_vector_storage_delete(tmp_path):
    storage = VectorStorage(tmp_path / "test_vectors.db")

    storage.save(1, "test-vector", "test", [1.0, 2.0])
    storage.delete(1)

    assert storage.load() == []


def test_vector_storage_updates_existing_id(tmp_path):
    storage = VectorStorage(tmp_path / "test_vectors.db")

    storage.save(1, "old", "test", [1.0, 2.0])
    storage.save(1, "updated", "new", [3.0, 4.0])

    records = storage.load()

    assert len(records) == 1
    assert records[0]["metadata"] == "updated"
    assert records[0]["category"] == "new"
    assert records[0]["embedding"] == [3.0, 4.0]
