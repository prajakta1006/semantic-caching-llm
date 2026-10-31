
"""Offline unit tests for MongoCacheStore."""

from src.models import CacheEntry
from src.mongo_cache_store import MongoCacheStore


class FakeCollection:
    """Minimal in-memory stand-in for a MongoDB collection."""

    def __init__(self):
        self.documents = {}

    def replace_one(self, filter_, document, upsert=False):
        key = filter_["_id"]
        if upsert or key in self.documents:
            self.documents[key] = dict(document)

    def find(self, filter_):
        return list(self.documents.values())

    def delete_one(self, filter_):
        self.documents.pop(filter_["_id"], None)

    def delete_many(self, filter_):
        if not filter_:
            self.documents.clear()
        elif "_id" in filter_ and "$nin" in filter_["_id"]:
            keep = set(filter_["_id"]["$nin"])
            self.documents = {
                key: value
                for key, value in self.documents.items()
                if key in keep
            }


def make_store():
    collection = FakeCollection()
    store = MongoCacheStore(collection=collection)
    return store, collection


def test_save_and_load_entry():
    store, collection = make_store()
    entry = CacheEntry(
        query="What is a binary search tree?",
        response="A tree-based search data structure.",
        embedding=[0.1, 0.2, 0.3],
        metadata={"access_count": 4, "created_at": 1234.0},
    )

    store.save_entry(entry)
    loaded = store.load_entries()

    assert len(loaded) == 1
    assert loaded[0].query == entry.query
    assert loaded[0].response == entry.response
    assert loaded[0].embedding == entry.embedding
    assert loaded[0].metadata == entry.metadata
    assert loaded[0].timestamp == entry.timestamp
    assert collection.documents[entry.query]["_id"] == entry.query


def test_saving_same_query_updates_existing_entry():
    store, collection = make_store()

    store.save_entry(
        CacheEntry(query="hello", response="old answer", embedding=[0.1])
    )
    store.save_entry(
        CacheEntry(query="hello", response="new answer", embedding=[0.2])
    )

    loaded = store.load_entries()

    assert len(loaded) == 1
    assert loaded[0].response == "new answer"
    assert loaded[0].embedding == [0.2]


def test_delete_entry():
    store, _ = make_store()
    store.save_entry(CacheEntry(query="hello", response="hi"))

    store.delete_entry("hello")

    assert store.load_entries() == []


def test_clear_entries():
    store, _ = make_store()
    store.save_entry(CacheEntry(query="hello", response="hi"))
    store.save_entry(CacheEntry(query="bye", response="goodbye"))

    store.clear()

    assert store.load_entries() == []


def test_entry_without_embedding_round_trips():
    store, _ = make_store()
    store.save_entry(
        CacheEntry(
            query="question",
            response="answer",
            embedding=None,
            metadata={"source": "test"},
        )
    )

    loaded = store.load_entries()

    assert len(loaded) == 1
    assert loaded[0].embedding is None
    assert loaded[0].metadata == {"source": "test"}
