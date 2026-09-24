# TigerGraph Agentic GraphRAG — Olympic Events Benchmark

Built for the TigerGraph Agentic GraphRAG Hackathon, September 2026.

---

## What We Built

A system that answers questions about Olympic events using three different retrieval architectures — RAG, GraphRAG, and Agentic GraphRAG — benchmarked head-to-head on 100 evaluation questions, with a full web interface that lets you run live queries and watch each pipeline work in real time.

The core question this project answers: **does graph-based retrieval actually beat vector search, and by how much?**

The short answer: yes, dramatically. GraphRAG achieves 85% exact match with zero LLM calls. RAG achieves 29% while spending ~1,856 tokens per question.

---

## The Three Pipelines

### RAG — the baseline

Vector similarity search over 2,951 Wikipedia articles. Embeds the question with `nomic-embed-text`, retrieves the top 5 most similar documents, passes them to `qwen3:4b`, and generates an answer. This is the standard approach most teams use.

Its weakness: aggregation questions need counts across 20+ events, but RAG only sees 5 documents. It gets 0% on aggregation and 0% on superlative questions.

### GraphRAG — structured traversal, no LLM

Instead of searching documents, GraphRAG queries a TigerGraph knowledge graph directly.

```
Question
  -> extract entities (year, season, sport, venue, date, threshold)
  -> TigerGraph REST++ vertex filter
  -> edge traversal (PREV/NEXT_EDITION for temporal, HELD_AT for multi-hop)
  -> read structured attribute (gold medalist, competitor count, nation count)
  -> return answer directly
```

No LLM involved. The graph encodes exactly what the questions ask about, so traversal replaces both retrieval and reasoning. Aggregation becomes a filter + count. Temporal questions become a two-hop edge traversal. Lookup is an exact attribute read.

Result: 85% exact match, 0 tokens, sub-second latency on most question types.

### Agentic GraphRAG — adaptive multi-step reasoning

An LLM plans the investigation, then executes it step by step using a set of graph and vector tools.

```
Question
  -> qwen3:4b produces a structured plan (year, season, sport, direction, threshold...)
  -> regex enrichment fills any plan fields the LLM missed
  -> adaptive tool execution:
       graph_filter, count_results, sort_results, follow_edge,
       infobox_lookup, venue_date_filter, vector_search
  -> synthesize answer from evidence chain
  -> if answer missing: retry with vector-augmented graph search
  -> if still missing: pure vector fallback
```

The agent evaluates evidence at each step and decides what to do next — it is not a fixed pipeline. It matches GraphRAG at 84% while handling edge cases that pure graph traversal misses.

---

## Benchmark Results

| Pipeline | Exact Match | Contains Gold | Avg LLM Tokens |
|---|---|---|---|
| RAG (baseline) | 29% | 43% | ~1,856/q |
| GraphRAG | **85%** | **85%** | **0** |
| Agentic GraphRAG | **84%** | **84%** | ~0/q |

### By Question Type

| Type | Questions | RAG | GraphRAG | Agentic |
|---|---|---|---|---|
| aggregation | 21 | 0% | 95.2% | 95.2% |
| lookup | 19 | 89.5% | 100% | 100% |
| superlative | 10 | 0% | 90.0% | 90.0% |
| temporal | 22 | 40.9% | 100% | 95.5% |
| multi_hop | 28 | 10.7% | 53.6% | 53.6% |
| **Overall** | **100** | **29%** | **85%** | **84%** |

RAG completely fails on aggregation and superlative — these require counting or ranking across the entire corpus, which top-5 retrieval cannot do. GraphRAG handles both perfectly because the graph stores structured counts per event. The only category where RAG is competitive is lookup (89.5%) — single-fact questions where the answer is likely in the top retrieved document.

---

## The Knowledge Graph

Hosted on TigerGraph Savanna 4.2.5, graph name: `OlympicsGraph`.

**Vertices**

| Type | Count | What it represents |
|---|---|---|
| OlympicEvent | 2,187 | One per event — "Athletics at the 2016 Summer Olympics – Men's 100 metres" |
| Medalist | 5,359 | Gold, silver, bronze medalists with name and country |
| Venue | 319 | Competition venues |

**Edges**

| Type | Meaning |
|---|---|
| GOLD_MEDAL / SILVER_MEDAL / BRONZE_MEDAL | Links an event to its medalists |
| HELD_AT | Links an event to its venue |
| PREV_EDITION / NEXT_EDITION | Temporal chain — links each event to the same event at the prior/next Olympics |

The PREV/NEXT_EDITION edges are what make temporal questions trivial. "Who won gold in event X at the Olympics immediately before 2010?" is two REST++ calls: find the anchor event, traverse one PREV_EDITION edge, read the gold medalist attribute.

---

## The Web Interface

A React + Vite frontend with five pages, all powered by a live FastAPI backend connected to TigerGraph.

### Dashboard

Benchmark overview. Shows exact match accuracy for all three pipelines with a live bar chart, accuracy breakdown by question type, and a side-by-side stat summary. Numbers pull from the actual results JSON — nothing hardcoded.

### Questions Explorer

Browse all 100 benchmark questions. Each question shows the gold answer and the predicted answer from all three pipelines side by side, with a pass/fail indicator. Filter by question type or pipeline accuracy.

### Live Query

Ask any Olympic question and watch the answer get computed in real time. Three pipeline tabs — switch between GraphRAG, Agentic, and RAG.

For GraphRAG: a full Cytoscape.js graph animation plays showing the traversal path. Signal dots travel along edges, traversed nodes light up in cyan, and the answer node turns gold at the end. After the animation, a collapsible Graph Query Trace panel shows every REST++ operation that ran — SOURCE, FILTER, EDGE, INFOBOX, RETURN — with the actual endpoint called.

For Agentic: a step-by-step tool call trace animates as the agent runs — each tool call appears with its input, output, and a status indicator.

For RAG: retrieved documents appear as ranked cards with staggered animation as each one loads.

A "Surprise me" button picks a random example question and fires it automatically. A copy button lets you grab the answer.

### Compare

Fires all three pipelines simultaneously against the same question and shows them side by side. A speed race bar tracks each pipeline's elapsed time independently as they run — bars only grow, never jump backward, using a log scale so fast pipelines (GraphRAG at ~3s) and slow ones (RAG at ~150s) are both visible. Each column shows that pipeline's visualization live. Once all three finish, answers appear with their elapsed times for direct comparison. The GraphRAG column also shows the Query Trace panel.

### Architecture

Pipeline architecture diagrams, graph schema, and live statistics pulled directly from TigerGraph (vertex counts, edge types). Accuracy percentages update from the live results API.

---

## Project Structure

```
├── pipelines/
│   ├── shared.py                # Corpus, EmbeddingIndex, llm_call, make_result
│   ├── graphrag_pipeline.py     # Pipeline 2: entity extraction + graph traversal
│   ├── agentic_pipeline.py      # Pipeline 3: LLM planner + adaptive tool execution
│   └── rag_pipeline.py          # Pipeline 1: vector similarity baseline
│
├── backend/
│   └── main.py                  # FastAPI: /api/query, /api/results, /api/graph
│
├── frontend/src/
│   ├── pages/
│   │   ├── Dashboard.jsx        # Benchmark overview
│   │   ├── QuestionExplorer.jsx # Browse all 100 questions
│   │   ├── LiveQuery.jsx        # Live query with per-pipeline visualization
│   │   ├── Compare.jsx          # Side-by-side parallel pipeline comparison
│   │   └── Architecture.jsx     # Diagrams and live graph stats
│   └── components/
│       ├── NetworkViz.jsx        # Cytoscape.js traversal animation
│       ├── AgenticViz.jsx        # Agent step-by-step trace
│       └── RAGViz.jsx            # Document retrieval cards
│
├── embed_documents.py           # Build 768-dim nomic embedding matrix (one-time)
├── load_graph.py                # Load vertices + edges into TigerGraph (one-time)
├── tg_client.py                 # TigerGraph REST++ client (JWT auth + retry)
├── compare_benchmarks.py        # Generate comparison table and HTML report
├── schema.gsql                  # TigerGraph schema definition
└── results/                     # Benchmark output JSON files
```

---

## Setup

### Prerequisites

- Python 3.12+
- Node.js 18+
- [Ollama](https://ollama.com/) running locally
- TigerGraph Savanna account — free tier at [savanna.tigergraph.com](https://savanna.tigergraph.com)

### Install

```bash
# Python dependencies
pip install requests numpy ollama python-dotenv fastapi uvicorn

# Frontend dependencies
cd frontend && npm install
```

### Configure

Create `.env` in the project root:

```
TG_HOST=https://your-instance.i.tgcloud.io
TG_GRAPH=OlympicsGraph
TG_SECRET=your_secret_here
TG_USERNAME=tigergraph
```

### One-time data setup

```bash
ollama pull nomic-embed-text
ollama pull qwen3:4b

python embed_documents.py   # build embedding index (~15 min)
python load_graph.py        # load vertices + edges into TigerGraph (~5 min)
```

### Run benchmarks

```bash
python -m pipelines.graphrag_pipeline
python -m pipelines.agentic_pipeline
python -m pipelines.rag_pipeline
python compare_benchmarks.py
```

### Run the web UI

```bash
# Terminal 1 — backend
python -m uvicorn backend.main:app --port 8000 --reload

# Terminal 2 — frontend
cd frontend && npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

---

## Models

| Model | Purpose | Size |
|---|---|---|
| `nomic-embed-text` | Document and query embeddings (768-dim) | 137M |
| `qwen3:4b` | Agentic planning, synthesis, RAG answers | 4B |

All inference runs locally via Ollama. GraphRAG uses zero LLM calls for most question types — the graph answers directly.

---

*TigerGraph Agentic GraphRAG Hackathon — September 2026*
