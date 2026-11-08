
from uuid import uuid4

from src.mongo_cache_store import MongoCacheStore
from src.models import CacheEntry


def main():
    store = MongoCacheStore(
        collection_name="cache_entries_smoke_test"
    )

    query = f"MongoDB persistence test {uuid4()}"

    try:
        original = CacheEntry(
            query=query,
            response="MongoDB persistence is working!",
            embedding=[0.1, 0.2, 0.3],
            metadata={"test": True},
        )

        store.save_entry(original)
        print("1. Entry saved to MongoDB.")

        # Create a second store instance to simulate restart.
        new_store = MongoCacheStore(
            collection_name="cache_entries_smoke_test"
        )

        try:
            restored_entries = new_store.load_entries()

            restored = next(
                entry
                for entry in restored_entries
                if entry.query == query
            )

            assert restored.response == original.response
            assert restored.embedding == original.embedding
            assert restored.metadata == original.metadata

            print("2. Entry successfully restored.")
            print("3. Response:", restored.response)
            print("4. Embedding:", restored.embedding)
            print("SUCCESS: Actual MongoDB persistence verified!")

        finally:
            new_store.close()

    finally:
        store.close()


if __name__ == "__main__":
    main()
