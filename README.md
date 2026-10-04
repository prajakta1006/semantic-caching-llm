# Semantic Caching for LLM - Cost-Benefit Cascade Flow

A DAA (Design and Analysis of Algorithms) course project that studies how semantic caching, routing decisions, and evaluation metrics can reduce repeated LLM cost and latency.

## Current Status

The TanTan foundation is implemented and tested. The project currently provides:

- A modular end-to-end pipeline coordinator.
- Clean interfaces for cache, cascade, LLM, and evaluator components.
- Dependency injection so teammates can plug in their modules later.
- A working exact-match mock cache.
- A deterministic mock LLM.
- A default cascade placeholder.
- A basic metrics recorder.
- CLI demo, lightweight web API, and existing UI.
- Unit/integration tests for the foundation.

The following teammate-owned algorithms are intentionally not implemented yet:

- Semantic embedding search.
- Cosine similarity or FAISS/vector indexing.
- Cost-benefit cascade optimization.
- Full benchmarking/evaluation analysis.

## Architecture

```text
User Query
    |
    v
Query Preprocessing
    |
    v
Cache.get(query, threshold)
    |
    +-- HIT --> Return cached response
    |
    +-- MISS
          |
          v
      Cascade.decide(query)
          |
          v
      LLM.generate(query, selected_model)
          |
          v
      Cache.put(query, response)
          |
          v
      Evaluator.record(...)
          |
          v
      Return response
```

## Team Responsibilities

| Person | Responsibility | Integration Point |
| --- | --- | --- |
| TanTan | Foundation, interfaces, pipeline, configuration, integration testing | `src/pipeline.py`, `src/interfaces.py`, `src/components.py`, `src/config.py` |
| Dhano | Real semantic cache, embeddings, similarity search, threshold behavior | Implement `CacheInterface` |
| Praj | Cost-benefit cascade and model/path routing | Implement `CascadeInterface` |
| Andi | Evaluation, benchmarks, metrics, experimental analysis | Implement `EvaluatorInterface` |

## Project Structure

```text
semantic-caching-llm/
|-- cache/
|   `-- .gitkeep
|-- src/
|   |-- __init__.py
|   |-- components.py      # Default/mock component implementations
|   |-- config.py          # Environment-backed settings
|   |-- interfaces.py      # Component contracts for teammates
|   |-- main.py            # CLI demo
|   |-- models.py          # Dataclasses used across components
|   |-- pipeline.py        # TanTan orchestration layer
|   `-- server.py          # Lightweight API/static UI server
|-- tests/
|   |-- __init__.py
|   |-- test_pipeline_integration.py
|   `-- test_setup.py
|-- ui/
|   |-- app.js
|   |-- index.html
|   `-- style.css
|-- requirements.txt
|-- run.py
`-- README.md
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

CLI demo:

```powershell
python run.py
```

Web UI:

```powershell
python run.py --ui
```

Then open:

```text
http://localhost:8000
```

Tests:

```powershell
python -m pytest
```

## Interfaces

The main contracts live in `src/interfaces.py`.

Cache:

```python
class CacheInterface:
    def get(self, query: str, threshold: float) -> CacheEntry | None: ...
    def put(self, query: str, response: str, embedding=None, metadata=None) -> None: ...
    def size(self) -> int: ...
```

Cascade:

```python
class CascadeInterface:
    def decide(self, request: QueryRequest, config: AppConfig) -> DecisionResult: ...
```

LLM:

```python
class LLMInterface:
    def generate(self, query: str, model: str | None = None) -> LLMResponse: ...
```

Evaluator:

```python
class EvaluatorInterface:
    def record(self, request: QueryRequest, response: PipelineResponse, latency_ms: float) -> None: ...
    def get_metrics(self) -> EvaluationMetrics: ...
```

The pipeline accepts injected components:

```python
pipeline = SemanticCachingPipeline(
    cache=YourCache(),
    cascade=YourCascade(),
    llm=YourLLM(),
    evaluator=YourEvaluator(),
)
```

## Configuration

Settings are read from environment variables with safe local defaults.

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Future embedding model target for Dhano |
| `SIMILARITY_THRESHOLD` | `0.85` | Cache hit threshold |
| `CASCADE_COST_THRESHOLD` | `0.05` | Future cascade cost setting for Praj |
| `CASCADE_QUALITY_THRESHOLD` | `0.75` | Future cascade quality setting for Praj |
| `CACHE_DIR` | `./cache` | Local cache directory |
| `LLM_PROVIDER` | `mock` | Future provider selector |
| `DEFAULT_LLM_MODEL` | `mock-llm-v1` | Default model name |
| `LLM_API_KEY_ENV` | `LLM_API_KEY` | Name of env var a future API LLM can read |
| `LLM_TIMEOUT_SECONDS` | `30` | Future API timeout |
| `DEBUG` | `false` | Debug flag |

No paid API key is required for the current demo. `LLM_PROVIDER=mock` is the default.

## Current Limitations

- `ExactMatchCache` is only an in-memory exact-match adapter.
- It does not generate embeddings.
- It does not compute cosine similarity.
- It does not use FAISS or a vector database.
- `DefaultCascade` always routes cache misses to the default mock LLM.
- `BasicEvaluator` records only simple session metrics.
- The UI remains a lightweight demo and was not redesigned.

## Handoff Notes

Dhano should implement `CacheInterface` in a new module such as `src/cache.py`. The pipeline will call only `get`, `put`, and `size`.

Praj should implement `CascadeInterface` in a new module such as `src/cascade.py`. Return `DecisionResult` with `metadata["selected_model"]` when selecting a model/path.

Andi should implement `EvaluatorInterface` in a new module such as `src/evaluator.py`. Use `record(...)` for telemetry and `get_metrics()` for reports.
