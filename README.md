# TigerGraph Agentic GraphRAG — Olympic Events Benchmark

Hackathon submission for TigerGraph Savanna, Round 1 (September 2026).

Benchmarks three knowledge-retrieval architectures against 100 evaluation questions on a dataset of 2,951 Olympic event Wikipedia articles, backed by a live TigerGraph knowledge graph.

---

## Results

| Pipeline | Exact Match | Contains Gold | Avg LLM Tokens |
|---|---|---|---|
| RAG (baseline) | 29% | 43% | ~1,856/q |
| GraphRAG | **85%** | **85%** | **0** |
| Agentic GraphRAG | **84%** | **84%** | **0** |

### By Question Type

| Type | N | RAG EM | GraphRAG EM | Agentic EM |
|---|---|---|---|---|
| aggregation | 21 | 0% | 95.2% | 95.2% |
| lookup | 19 | 89.5% | 100% | 100% |
| multi_hop | 28 | 10.7% | 53.6% | 53.6% |
| superlative | 10 | 0% | 90.0% | 90.0% |
| temporal | 22 | 40.9% | 100% | 95.5% |
| **Overall** | **100** | **29%** | **85%** | **84%** |

GraphRAG achieves 85% exact match with zero LLM calls. RAG fails completely on aggregation and superlative question types because they require counting or ranking across the full event corpus, not just the top-5 retrieved documents.

---

## How It Works

### Pipeline 1 — RAG (Baseline)

```
Question
  -> nomic-embed-text (768-dim embedding)
  -> cosine similarity over 2,951 documents
  -> top-5 documents (truncated to 800 chars each)
  -> qwen3:4b LLM
  -> answer
```

### Pipeline 2 — GraphRAG

```
Question
  -> regex entity extraction (year, season, sport, venue, date, threshold)
  -> TigerGraph REST++ vertex filter
  -> edge traversal (PREV/NEXT_EDITION for temporal, HELD_AT for multi-hop)
  -> infobox field parsing
  -> direct structured answer (no LLM)
```

Question-type routing:
- **Aggregation:** filter by sport/year/season, count events where competitors > threshold
- **Superlative:** filter, sort by competitor or nation count
- **Temporal:** anchor event lookup, traverse PREV/NEXT_EDITION edge, read gold medalist
- **Multi-hop:** filter by year, match venue + date, read gold medalist
- **Lookup:** title keyword overlap, read attribute directly

### Pipeline 3 — Agentic GraphRAG

```
Question
  -> qwen3:4b LLM produces structured plan (year, season, sport, direction, threshold...)
  -> regex enrichment fills any missing plan fields
  -> adaptive tool execution across: graph_filter, count_results, sort_results,
     follow_edge, infobox_lookup, venue_date_filter, vector_search
  -> answer synthesized from evidence chain
  -> if answer missing: retry with vector-augmented graph search
  -> if still missing: pure vector fallback
```

Handles edge cases GraphRAG misses through multi-step reasoning, at the cost of higher latency.

---

## Graph Schema (TigerGraph Savanna 4.2.5)

**Graph name:** OlympicsGraph

### Vertex Types

| Type | Count | Key Attributes |
|---|---|---|
| OlympicEvent | 2,187 | title, year, season, sport, event, venue_name, date_str, competitors, nations, doc_id |
| Medalist | 5,359 | name, country |
| Venue | 319 | name |

### Edge Types

| Edge | From | To |
|---|---|---|
| GOLD_MEDAL / SILVER_MEDAL / BRONZE_MEDAL | OlympicEvent | Medalist |
| HELD_AT | OlympicEvent | Venue |
| PREV_EDITION / NEXT_EDITION | OlympicEvent | OlympicEvent |

---

## Dataset

- **Corpus:** `corpus.jsonl` — 2,951 Wikipedia articles covering Olympic events (infobox + article text)
- **Questions:** `eval_public.jsonl` — 100 evaluation questions with gold answers across 5 question types
- **Embeddings:** 768-dimensional vectors from `nomic-embed-text`, precomputed and stored locally

---

## Project Structure

```
GraphRAG hackathon/
├── .env                         # TG_HOST, TG_SECRET, TG_GRAPH (not committed)
├── embed_documents.py           # Build 768-dim embedding matrix (one-time)
├── load_graph.py                # Load vertices + edges into TigerGraph (one-time)
├── tg_client.py                 # TigerGraph REST++ client (JWT auth + retry)
├── compare_benchmarks.py        # Generate comparison table and HTML report
│
├── pipelines/
│   ├── shared.py                # Corpus, EmbeddingIndex, llm_call, make_result
│   ├── rag_pipeline.py          # Pipeline 1: RAG
│   ├── graphrag_pipeline.py     # Pipeline 2: GraphRAG
│   └── agentic_pipeline.py      # Pipeline 3: Agentic GraphRAG
│
├── backend/
│   └── main.py                  # FastAPI — /api/query, /api/results, /api/graph
│
├── frontend/
│   ├── src/pages/
│   │   ├── Dashboard.jsx        # Benchmark overview with live accuracy stats
│   │   ├── QuestionExplorer.jsx # Browse all 100 benchmark questions + answers
│   │   ├── LiveQuery.jsx        # Live query with graph traversal animation
│   │   ├── Compare.jsx          # Side-by-side pipeline comparison with speed race
│   │   └── Architecture.jsx     # Pipeline architecture diagrams
│   └── src/components/
│       ├── NetworkViz.jsx        # Cytoscape.js graph traversal visualization
│       ├── AgenticViz.jsx        # Agentic tool-call step visualization
│       └── RAGViz.jsx            # RAG document retrieval visualization
│
└── results/
    ├── graphrag_results.json
    ├── agentic_results.json
    └── rag_results.json
```

---

## Setup

### Prerequisites

- Python 3.12+
- Node.js 18+
- [Ollama](https://ollama.com/) running locally with `nomic-embed-text` and `qwen3:4b` pulled
- TigerGraph Savanna account — free tier at [savanna.tigergraph.com](https://savanna.tigergraph.com)

### 1. Python dependencies

```bash
pip install requests numpy ollama python-dotenv fastapi uvicorn
```

### 2. Frontend dependencies

```bash
cd frontend
npm install
```

### 3. Environment variables

Create a `.env` file in the project root:

```
TG_HOST=https://your-instance.i.tgcloud.io
TG_GRAPH=OlympicsGraph
TG_SECRET=your_secret_here
TG_USERNAME=tigergraph
```

### 4. One-time data setup

```bash
# Pull Ollama models
ollama pull nomic-embed-text
ollama pull qwen3:4b

# Build embedding index (~15 min)
python embed_documents.py

# Load graph data into TigerGraph (~5 min)
python load_graph.py
```

---

## Running

### Run benchmarks

```bash
python -m pipelines.graphrag_pipeline
python -m pipelines.agentic_pipeline
python -m pipelines.rag_pipeline

# Generate comparison report
python compare_benchmarks.py
```

### Run the web UI

Start the backend (from project root):

```bash
python -m uvicorn backend.main:app --port 8000 --reload
```

Start the frontend (in a second terminal):

```bash
cd frontend
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

---

## Web UI

Five pages:

| Page | Description |
|---|---|
| Dashboard | Live benchmark results, accuracy by question type |
| Questions | Browse all 100 benchmark questions with answers from each pipeline |
| Live Query | Ask any Olympic question, watch graph traversal animate in real time |
| Compare | Fire all 3 pipelines simultaneously, see speed race and trace panel |
| Architecture | Pipeline diagrams and graph schema with live stats |

The Compare page fires all three pipelines in parallel against the same question and shows each pipeline's elapsed time independently.

---

## Models

| Model | Purpose | Parameters |
|---|---|---|
| `nomic-embed-text` | Document and query embeddings (768-dim) | 137M |
| `qwen3:4b` | LLM for RAG answers, agentic planning, synthesis | 4B |

All inference runs locally via Ollama. GraphRAG uses zero LLM calls for most question types.

---

*Built for the TigerGraph Agentic GraphRAG Hackathon, September 2026.*
