# Semantic Caching for LLM — Cost-Benefit Cascade Flow

A DAA (Design and Analysis of Algorithms) project exploring algorithmic trade-offs between semantic similarity caching, cost-benefit cascading, and inference latency for Large Language Model queries.

---

## 1. Problem Being Solved

Large Language Models (LLMs) are computationally expensive, exhibit high latency, and incur per-token financial costs for repeated or semantically identical queries. 

Traditional exact-match caching fails when queries differ slightly in wording (e.g., *"What is machine learning?"* vs *"Can you explain machine learning?"*). 

This project implements:
1. **Semantic Caching:** Vector similarity search to match and reuse cached responses for semantically equivalent queries.
2. **Cost-Benefit Cascade Flow:** An algorithmic decision engine that evaluates query complexity, cache similarity confidence, and cost thresholds to determine whether to return cached results, invoke a lightweight fallback, or route to a full LLM.
3. **Evaluation Framework:** Empirical metrics tracking hit rates, latency reductions, and cost savings under various threshold settings.

---

## 2. High-Level Flow Architecture

```
                 User Query
                     │
                     ▼
             Query Processing
                     │
                     ▼
           Semantic Cache Check
                     │
              [ Cache Hit? ]
             ┌───────┴───────┐
      YES    │               │    NO
             ▼               ▼
     Return Cached   Cascade / Cost-Benefit
       Response            Decision
                             │
                             ▼
                        LLM / API Call
                             │
                             ▼
                    Store Result in Cache
                             │
                             ▼
                      Return Response
                             │
                             ▼
                    Evaluation / Metrics
```

---

## 3. Team Responsibilities & Code Locations

This project is built collaboratively with clean modular boundaries:

| Teammate | Role | Responsibility | Module Location / Extension Point |
| :--- | :--- | :--- | :--- |
| **Person 1: TanTan** | Project Lead & Core Foundation | Repository setup, configuration, data models, integration pipeline, and CLI runner | `src/config.py`, `src/models.py`, `src/pipeline.py`, `run.py` |
| **Person 2: Dhano** | Semantic Cache Module | Vector embeddings (`sentence-transformers`), similarity indexing (FAISS / Cosine), cache eviction (LRU/LFU) | Implement `CacheInterface` in `src/cache.py` (plugs into `src/pipeline.py`) |
| **Person 3: Praj** | Cascade / Decision Engine | Cost-benefit algorithm, query complexity scoring, and dynamic routing logic | Implement `DecisionInterface` in `src/cascade.py` (plugs into `src/pipeline.py`) |
| **Person 4: Andi** | Evaluation & Benchmarking | Metrics collection, benchmark datasets, cost-reduction analysis, and automated test suite | Implement `EvaluatorInterface` in `src/evaluator.py`, `tests/` |

---

## 4. Project Structure

```
semantic-caching-llm/
│
├── .venv/                   # Virtual environment (ignored by git)
├── cache/                   # Local cache directory
│   └── .gitkeep
│
├── src/
│   ├── __init__.py          # Package initializer
│   ├── config.py            # Centralized settings & environment variables
│   ├── models.py            # Typed dataclasses (Request, CacheEntry, Decision, Metrics)
│   ├── pipeline.py          # Core pipeline coordinator & abstract interfaces
│   └── main.py              # CLI demo execution logic
│
├── tests/
│   ├── __init__.py
│   └── test_setup.py        # Foundation & environment verification tests
│
├── requirements.txt         # Core dependencies
├── .gitignore               # Git ignore rules
├── README.md                # Project documentation
└── run.py                   # Root execution script
```

---

## 5. Getting Started (Windows PowerShell)

### Step 1: Clone Repository
```powershell
git clone <repository-url>
cd semantic-caching-llm
```

### Step 2: Create & Activate Virtual Environment
```powershell
# Create virtual environment (Python 3.10+)
python -m venv .venv

# Activate in Windows PowerShell
.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then run the activate command again).*

### Step 3: Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 6. How to Run

### Run the Web UI (Modern Aesthetic Dashboard)
```powershell
python run.py --ui
```
*Opens `http://localhost:8000` with the custom modern AI assistant UI interface.*

### Run the CLI Demo
```powershell
python run.py
```

### Run Unit Tests
```powershell
python -m pytest
```
Or for verbose test logs:
```powershell
python -m pytest -v
```

---

## 7. Configuration (`src/config.py`)

Settings can be customized via environment variables or modified directly in `src/config.py`:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | HuggingFace sentence embedding model |
| `SIMILARITY_THRESHOLD` | `0.85` | Cosine similarity threshold for cache hits |
| `CASCADE_COST_THRESHOLD`| `0.05` | Maximum threshold before escalating query |
| `CACHE_DIR` | `./cache` | Local directory for persisted cache data |
| `DEFAULT_LLM_MODEL` | `mock-llm-v1` | Target LLM model name |
| `DEBUG` | `false` | Enable verbose debug logging |

---

## 8. Troubleshooting `sentence-transformers` & `scipy` on Windows

If you encounter DLL or import errors such as:
`ImportError: DLL load failed while importing _biasedurn: An Application Control policy has blocked this file`

### Resolution:
1. **Scipy Version Compatibility:**
   On Python 3.14+ on Windows with Application Control or Smart App Control, ensure `scipy` is pinned to `1.17.1`:
   ```powershell
   pip install scipy==1.17.1
   ```
2. **Verify Sentence Transformers:**
   ```powershell
   python -c "from sentence_transformers import SentenceTransformer; print('Sentence Transformers OK')"
   ```
3. **Unblock Downloaded DLLs (if blocked by Windows):**
   ```powershell
   Get-ChildItem -Path .venv\Lib\site-packages\scipy -Recurse -Filter *.pyd | Unblock-File
   ```

---

## 9. How Teammates Should Add Code

### For Dhano (`CacheInterface`):
1. Create `src/cache.py`.
2. Inherit from `src.pipeline.CacheInterface`.
3. Implement `lookup(query, threshold)`, `insert(query, response, embedding, metadata)`, and `size()`.
4. Inject your cache class into `SemanticCachingPipeline(cache=YourCache())`.

### For Praj (`DecisionInterface`):
1. Create `src/cascade.py`.
2. Inherit from `src.pipeline.DecisionInterface`.
3. Implement `decide(request, cache_match, config) -> DecisionResult`.
4. Inject into `SemanticCachingPipeline(decision_engine=YourDecisionEngine())`.

### For Andi (`EvaluatorInterface`):
1. Create `src/evaluator.py`.
2. Inherit from `src.pipeline.EvaluatorInterface`.
3. Implement `record(request, response, latency_ms)` and `get_metrics()`.
4. Add automated benchmark tests in `tests/`.
