# Semantic Caching for LLM Applications

A Design and Analysis of Algorithms (DAA) project that reduces repeated LLM work by reusing responses to exact or semantically similar questions. The project combines an embedding based cache, cache replacement and expiration policies, optional MongoDB persistence, a pipeline, MockLLM and Google Gemini support, and evaluation tools.

## Problem and objective

LLM requests can add latency and incur provider costs even when users repeat a question or ask a close paraphrase. This project stores query/response pairs and their embeddings, then compares each new query with stored vectors. A sufficiently similar entry is returned without an LLM request. The similarity threshold, cache capacity, LRU policy and TTL determine the tradeoff between reuse and freshness.

## Architecture and request flow

```text
User query
  -> preprocess (trim whitespace)
  -> CacheOptimizer: remove expired entries
  -> SemanticCache: encode query and compare cosine similarity
       -> score >= SIMILARITY_THRESHOLD: return cached response
       -> miss: LLMService selects MockLLM or GeminiLLM
  -> store a successful response in SemanticCache / MongoDB
  -> update evaluator metrics and return response
```

The pipeline depends on `CacheInterface`, `LLMInterface` and `EvaluatorInterface`. The modules are composed in `run.py`; the HTTP adapter is in `src/server.py`. Cache-write failures are recorded while still returning a successfully generated LLM response. Failed LLM responses are not cached.

## Cache algorithms and DAA complexity

- **Semantic matching:** embeddings are compared with cosine similarity. For n cached entries and vector dimension d, a lookup scans vectors in O(n·d) time and uses O(d) temporary space. Embedding generation has model-dependent cost in addition to the scan.
- **Exact-query hashing:** `SemanticCache` keeps a dictionary keyed by query for expected O(1) exact lookup. An update also changes the ordered list used for semantic scans, so replacement currently costs O(n).
- **LRU:** `CacheOptimizer` uses an `OrderedDict`. A successful hit moves the entry to the most recently used end in O(1), and selecting the least recently used victim is O(1). Removing that entry from the semantic cache list costs O(n).
- **TTL:** expiration is checked by scanning optimizer metadata, O(n) per check. Expired records are removed from the cache and persistence store.
- **Frequency:** successful cache hits increment and persist `access_count`. This is available as usage metadata; the active eviction policy is LRU, not LFU.
- **Capacity:** insertion evicts least recently used entries until size is within `max_size`. The cache uses O(n·d + response data) storage. There is no heap or FAISS/vector index.

An identical query string replaces its prior response. Distinct strings remain separate entries even if semantically similar; similarity controls lookup, not insertion deduplication.

## Persistence

`MongoCacheStore` uses the existing collection format and exact query as MongoDB `_id`. It supports save/update, load, delete, clear and bulk synchronization. The store is optional when constructing `SemanticCache` directly; the interactive `run.py` application currently initializes MongoDB and therefore requires a connection URI. Offline persistence tests inject fake collections.

Configuration is loaded from environment variables and a project-root `.env` file. Copy `.env.example` to `.env`, then set:

- `MONGODB_URI` to the Atlas connection string.
- `MONGODB_DATABASE` (default `semantic_cache_db`).
- `MONGODB_COLLECTION` (default `cache_entries`).

Never commit `.env` or put credentials in source. The tracked `.env.example` contains placeholders only.

## LLM providers

`LLMService` selects a provider using `LLM_PROVIDER`:

- `mock` uses deterministic `MockLLM` and needs no API key. This is the recommended offline/test setting.
- `gemini` uses `GeminiLLM` and reads the key from the environment variable named by `LLM_API_KEY_ENV` (default `LLM_API_KEY`). It uses bounded timeouts, retries transient failures, and can try a fallback model. Model names are configurable with `DEFAULT_LLM_MODEL`.

The Gemini implementation is tested with mocked HTTP responses. No live Gemini request was needed by the offline test suite. Provider pricing is not calculated by `GeminiLLM`; its returned cost is currently zero unless another integration supplies cost data. MockLLM costs are simulated values.

## Evaluation

- `evaluation/benchmark.py` compares baseline and cached runs on `datasets/evaluation_queries.json` and saves CSV/JSON results. It uses a simulated LLM; embedding initialization uses the configured SentenceTransformer model.
- `evaluation/benchmark_delayed.py` measures the effect of a deliberately delayed mock LLM.
- `evaluation/answer_quality.py` assesses answer reuse quality on the supplied evaluation dataset.
- `evaluation/threshold_comparison.py` compares configured similarity thresholds.
- `src/evaluator.py` records runtime hits, misses, latency, and estimated cost metrics.

Result files in `evaluation/results/` are retained artifacts from prior runs. They are not automatically regenerated by the unit test suite. Benchmark numbers are simulated/estimated where the scripts use MockLLM and should not be read as live provider measurements.

## Installation and configuration

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` for the intended mode. For offline work, use `LLM_PROVIDER=mock`; to run the interactive demo, also configure `MONGODB_URI`. Set `LLM_PROVIDER=gemini` and a real key only when you intend to make Gemini requests. The embedding model may be downloaded the first time it is loaded.

Relevant settings include `EMBEDDING_MODEL_NAME`, `SIMILARITY_THRESHOLD`, `LLM_PROVIDER`, `DEFAULT_LLM_MODEL`, `LLM_API_KEY_ENV`, `LLM_TIMEOUT_SECONDS`, `MONGODB_URI`, `MONGODB_DATABASE`, and `MONGODB_COLLECTION`.

## Run the application and tests

Start the interactive CLI (requires configured MongoDB):

```powershell
python run.py
```

Start the development web UI/API (runs with the configured LLM; the server module has no MongoDB cache wired by default):

```powershell
python -m src.server
```

Run offline unit and integration tests without MongoDB or Gemini credentials:

```powershell
python -m pytest -p no:cacheprovider tests
```

The root-level `test_mongodb_connection.py` is a separate live connectivity check and requires `MONGODB_URI`; it is not part of the offline suite.

## Current limitations

- Semantic lookup scans all cached embeddings; no approximate nearest-neighbor index is implemented.
- LRU is the eviction policy. Frequency is tracked but not used for LFU eviction.
- TTL expiration checks scan current cache metadata.
- Gemini and Atlas were not live-tested as part of the offline suite; their external availability depends on credentials and service configuration.
- Gemini cost is not estimated by the current client.
