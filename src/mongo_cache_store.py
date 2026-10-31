
"""MongoDB persistence layer for semantic cache entries."""

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from pymongo import MongoClient, ReplaceOne
from pymongo.collection import Collection

from src.models import CacheEntry


# Load environment settings from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class MongoCacheStore:
    """Persist cache entries in MongoDB Atlas."""

    def __init__(
        self,
        uri: Optional[str] = None,
        database_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        collection: Optional[Collection] = None,
    ) -> None:
        self._client = None

        if collection is not None:
            # Dependency injection makes unit testing possible offline.
            self.collection = collection
        else:
            uri = uri or os.getenv("MONGODB_URI")
            database_name = (
                database_name
                or os.getenv("MONGODB_DATABASE", "semantic_cache_db")
            )
            collection_name = (
                collection_name
                or os.getenv("MONGODB_COLLECTION", "cache_entries")
            )

            if not uri:
                raise ValueError(
                    "MONGODB_URI is missing. Check your .env file."
                )

            self._client = MongoClient(
                uri,
                serverSelectionTimeoutMS=10000,
            )
            self.collection = self._client[
                database_name
            ][collection_name]

    @staticmethod
    def _to_document(entry: CacheEntry) -> dict[str, Any]:
        """Convert a CacheEntry to a MongoDB document."""
        return {
            "_id": entry.query,
            "query": entry.query,
            "response": entry.response,
            "last_accessed": entry.last_accessed,
            "access_count": entry.access_count,
            "embedding": (
                list(entry.embedding)
                if entry.embedding is not None
                else None
            ),
            "similarity_score": entry.similarity_score,
            "timestamp": entry.timestamp,
            "metadata": dict(entry.metadata),
        }

    @staticmethod
    def _from_document(document: dict[str, Any]) -> CacheEntry:
        """Convert a MongoDB document to a CacheEntry."""
        return CacheEntry(
            query=document["query"],
            response=document["response"],
            embedding=document.get("embedding"),
            similarity_score=document.get("similarity_score", 0.0),
            last_accessed=document.get("last_accessed"),
            access_count=document.get("access_count", 0),
            timestamp=document["timestamp"],
            metadata=document.get("metadata", {}),
        )

    def save_entry(self, entry: CacheEntry) -> None:
        """Insert or replace an entry with the same exact query."""
        document = self._to_document(entry)
        self.collection.replace_one(
            {"_id": entry.query},
            document,
            upsert=True,
        )

    def load_entries(self) -> list[CacheEntry]:
        """Load all persisted entries."""
        documents = self.collection.find({})
        return [self._from_document(doc) for doc in documents]

    def delete_entry(self, query: str) -> None:
        """Delete an entry by its exact query."""
        self.collection.delete_one({"_id": query})

    def clear(self) -> None:
        """Delete all cache entries in this collection."""
        self.collection.delete_many({})

    def save_entries(self, entries: list[CacheEntry]) -> None:
        """Replace the stored cache contents with the supplied entries."""
        if not entries:
            self.clear()
            return

        operations = [
            ReplaceOne(
                {"_id": entry.query},
                self._to_document(entry),
                upsert=True,
            )
            for entry in entries
        ]

        self.collection.bulk_write(operations, ordered=True)

        # Remove persisted entries that are no longer in the cache.
        current_queries = [entry.query for entry in entries]
        self.collection.delete_many(
            {"_id": {"$nin": current_queries}}
        )

    def close(self) -> None:
        """Close the MongoDB connection if this store owns it."""
        if self._client is not None:
            self._client.close()
            self._client = None
