/**
 * SemCache AI — Frontend Logic & Pipeline Client
 * Connects to backend API or simulates pipeline client-side seamlessly
 */

// State
let sessionQueries = [];
let cacheStore = [
  {
    query: "what is machine learning?",
    response: "Machine learning is a field of AI focusing on building applications that learn from data and improve accuracy over time without being explicitly programmed.",
    simScore: 1.0,
    hits: 2,
    timestamp: new Date().toLocaleTimeString()
  }
];

let metrics = {
  totalQueries: 0,
  cacheHits: 0,
  cacheMisses: 0,
  hitRate: 0,
  costSaved: 0.0,
  totalLatency: 0.0
};

// Elements
const promptInput = document.getElementById("prompt-input");
const streamContainer = document.getElementById("stream-container");
const streamEmpty = document.getElementById("stream-empty");
const cacheTableBody = document.getElementById("cache-table-body");

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  renderCacheTable();
  updateMetricsDisplay();
  
  // Shortcut ⌘ K / Ctrl K to focus search
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "k") {
      e.preventDefault();
      document.getElementById("global-search")?.focus();
    }
  });
});

// Switch Views
function switchView(viewName) {
  document.querySelectorAll(".nav-item").forEach(item => {
    item.classList.toggle("active", item.getAttribute("data-view") === viewName);
  });

  const views = ["dashboard", "cache", "metrics", "team", "history"];
  views.forEach(v => {
    const el = document.getElementById(`view-${v}`);
    if (el) {
      el.classList.toggle("view-hidden", v !== viewName);
    }
  });

  if (viewName === "cache") renderCacheTable();
  if (viewName === "metrics") updateMetricsDisplay();
}

// Workspace filter pill click
function filterWorkspace(type, buttonEl) {
  document.querySelectorAll(".workspace-pills .pill").forEach(p => p.classList.remove("active"));
  buttonEl.classList.add("active");

  const templatesGrid = document.getElementById("templates-grid");
  if (!templatesGrid) return;

  if (type === "cache") {
    switchView("cache");
  } else if (type === "cascade") {
    switchView("team");
  } else if (type === "benchmark") {
    switchView("metrics");
  } else {
    switchView("dashboard");
  }
}

// Global search filter
function handleGlobalSearch(e) {
  const q = e.target.value.toLowerCase().trim();
  if (!q) return;

  const cards = document.querySelectorAll(".template-card");
  cards.forEach(card => {
    const text = card.textContent.toLowerCase();
    card.style.display = text.includes(q) ? "flex" : "none";
  });
}

// Execute query from template
function runTemplateQuery(queryText) {
  if (promptInput) {
    promptInput.value = queryText;
    executePipelineQuery(queryText);
  }
}

// Form submit
function handleFormSubmit(e) {
  e.preventDefault();
  const q = promptInput.value.trim();
  if (!q) return;
  executePipelineQuery(q);
}

// Reset chat
function resetChat() {
  if (streamContainer) streamContainer.innerHTML = "";
  if (streamEmpty) streamEmpty.style.display = "block";
  if (promptInput) promptInput.value = "";
  switchView("dashboard");
}

// Core Pipeline Execution
async function executePipelineQuery(query) {
  if (streamEmpty) streamEmpty.style.display = "none";

  // Append User message
  const userBubble = document.createElement("div");
  userBubble.className = "query-bubble-user";
  userBubble.textContent = query;
  streamContainer.appendChild(userBubble);
  promptInput.value = "";

  // Loading Placeholder
  const loadingCard = document.createElement("div");
  loadingCard.className = "response-card-ai";
  loadingCard.innerHTML = `<div style="display:flex;align-items:center;gap:10px;color:#64748b;">
    <span class="loading-spinner">⚡</span> Evaluating Semantic Cache & Cascade Routing...
  </div>`;
  streamContainer.appendChild(loadingCard);
  streamContainer.scrollTop = streamContainer.scrollHeight;

  const startTime = performance.now();

  try {
    // Attempt backend API call if server is running
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query })
    });

    if (response.ok) {
      const data = await response.json();
      renderPipelineResponse(loadingCard, data, performance.now() - startTime);
      return;
    }
  } catch (err) {
    // If running client-only static file, execute smart browser-side simulation
  }

  // Client Simulation fallback
  setTimeout(() => {
    const simulated = simulatePipeline(query);
    const latency = performance.now() - startTime;
    renderPipelineResponse(loadingCard, simulated, latency);
  }, 350);
}

// Render Response Card
function renderPipelineResponse(cardEl, data, clientLatency) {
  const isHit = data.source === "cache";
  const simScore = data.similarity_score !== null ? (data.similarity_score * 100).toFixed(1) + "%" : "N/A";
  const route = data.decision?.route || (isHit ? "CACHE_HIT" : "DIRECT_LLM");
  const latency = data.latency_ms ? data.latency_ms.toFixed(2) : clientLatency.toFixed(2);

  // Update telemetry metrics
  metrics.totalQueries++;
  if (isHit) {
    metrics.cacheHits++;
    metrics.costSaved += 0.002;
  } else {
    metrics.cacheMisses++;
  }
  metrics.hitRate = (metrics.cacheHits / metrics.totalQueries) * 100;
  metrics.totalLatency += parseFloat(latency);

  updateMetricsDisplay();
  renderCacheTable();

  cardEl.innerHTML = `
    <!-- Top Step Breakdown Pill -->
    <div class="pipeline-breakdown">
      <div class="breakdown-item">
        <span class="breakdown-label">Cache Status</span>
        <span class="breakdown-val">
          <span class="${isHit ? 'pill-hit' : 'pill-miss'}">${isHit ? 'HIT (Dhano Cache)' : 'MISS (LLM Route)'}</span>
        </span>
      </div>
      <div class="breakdown-item">
        <span class="breakdown-label">Similarity</span>
        <span class="breakdown-val">${simScore}</span>
      </div>
      <div class="breakdown-item">
        <span class="breakdown-label">Cascade Decision</span>
        <span class="breakdown-val">${route}</span>
      </div>
      <div class="breakdown-item">
        <span class="breakdown-label">Latency</span>
        <span class="breakdown-val">⚡ ${latency} ms</span>
      </div>
    </div>

    <!-- AI Output Content -->
    <div class="response-text-content">
      ${data.response_text}
    </div>
  `;

  streamContainer.scrollTop = streamContainer.scrollHeight;
}

// Client simulation engine when static
function simulatePipeline(query) {
  const cleanQ = query.toLowerCase().trim();
  
  // Check exact or partial match in cacheStore
  let matched = cacheStore.find(item => 
    item.query === cleanQ || 
    (cleanQ.includes("machine learning") && item.query.includes("machine learning"))
  );

  if (matched) {
    matched.hits++;
    return {
      query: query,
      response_text: matched.response,
      source: "cache",
      similarity_score: matched.query === cleanQ ? 1.0 : 0.93,
      decision: { route: "CACHE_HIT", estimated_cost: 0.0, confidence: 0.93 },
      latency_ms: 0.6
    };
  }

  // Cache Miss -> Generate mock LLM answer & store
  const newAnswer = cleanQ.includes("semantic caching")
    ? "Semantic caching stores query embeddings in a vector index to return cached outputs for semantically identical questions, cutting token costs and server latency."
    : `Synthesized LLM response for: "${query}". (Generated by foundation mock pipeline; Dhano/Praj/Andi modules plug in seamlessly).`;

  cacheStore.unshift({
    query: cleanQ,
    response: newAnswer,
    simScore: 1.0,
    hits: 1,
    timestamp: new Date().toLocaleTimeString()
  });

  return {
    query: query,
    response_text: newAnswer,
    source: "llm",
    similarity_score: 0.0,
    decision: { route: "DIRECT_LLM", estimated_cost: 0.002, confidence: 1.0 },
    latency_ms: 14.2
  };
}

// Update cache table view
function renderCacheTable() {
  if (!cacheTableBody) return;
  if (cacheStore.length === 0) {
    cacheTableBody.innerHTML = `<tr><td colspan="5" class="text-center">Cache is currently empty.</td></tr>`;
    return;
  }

  cacheTableBody.innerHTML = cacheStore.map(item => `
    <tr>
      <td><strong>${item.query}</strong></td>
      <td style="max-width:300px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${item.response}</td>
      <td><span class="tag-badge hit-badge">${(item.simScore * 100).toFixed(0)}%</span></td>
      <td><strong>${item.hits}</strong></td>
      <td><span style="color:#94a3b8; font-size:11px;">${item.timestamp}</span></td>
    </tr>
  `).join("");
}

// Update Metrics Display
function updateMetricsDisplay() {
  const totalEl = document.getElementById("metric-total");
  const hitRateEl = document.getElementById("metric-hit-rate");
  const latencyEl = document.getElementById("metric-latency");
  const costSavedEl = document.getElementById("metric-cost-saved");
  const sidebarCredits = document.getElementById("sidebar-credits-saved");
  const sidebarCacheStat = document.getElementById("sidebar-cache-stat");

  if (totalEl) totalEl.textContent = metrics.totalQueries;
  if (hitRateEl) hitRateEl.textContent = `${metrics.hitRate.toFixed(1)}%`;
  if (latencyEl) {
    const avg = metrics.totalQueries > 0 ? (metrics.totalLatency / metrics.totalQueries).toFixed(2) : "0.00";
    latencyEl.textContent = `${avg} ms`;
  }
  if (costSavedEl) costSavedEl.textContent = `$${metrics.costSaved.toFixed(3)}`;
  if (sidebarCredits) sidebarCredits.textContent = `$${metrics.costSaved.toFixed(3)}`;
  if (sidebarCacheStat) sidebarCacheStat.textContent = `${metrics.cacheHits} Hits • 85% Sim Threshold`;
}

// Clear Cache
function clearCacheStore() {
  cacheStore = [];
  renderCacheTable();
  alert("Cache memory cleared.");
}

function showTeamModal() {
  switchView("team");
}

function showHelpModal() {
  alert("SemCache AI: DAA Semantic Caching & Cascade Flow\n\n• Person 1 (TanTan): Core Pipeline & UI\n• Person 2 (Dhano): Vector Cache\n• Person 3 (Praj): Cascade Algorithm\n• Person 4 (Andi): Evaluation Metrics\n\nRun 'python run.py --ui' to start the local backend server.");
}

function openAddTemplateModal() {
  const custom = prompt("Enter a sample query for your custom template:");
  if (custom) runTemplateQuery(custom);
}
