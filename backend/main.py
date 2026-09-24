"""
FastAPI backend — serves benchmark results + live query endpoint.
Run: uvicorn backend.main:app --reload --port 8000
"""
import json
import re
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

app = FastAPI(title="Olympic GraphRAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

RESULTS_DIR = ROOT / "results"

# ── Singleton pipeline instances (lazy init, cached across requests) ──────────
_pipelines: dict = {}

def get_pipeline(name: str):
    if name not in _pipelines:
        if name == "graphrag":
            from pipelines.graphrag_pipeline import GraphRAGPipeline
            _pipelines[name] = GraphRAGPipeline()
        elif name == "agentic":
            from pipelines.agentic_pipeline import AgenticGraphRAGPipeline
            _pipelines[name] = AgenticGraphRAGPipeline()
        elif name == "rag":
            from pipelines.rag_pipeline import RAGPipeline
            _pipelines[name] = RAGPipeline()
        else:
            raise ValueError(f"Unknown pipeline: {name}")
    return _pipelines[name]


_OLYMPIC_KEYWORDS = {
    "olympic", "olympics", "gold", "silver", "bronze", "medal", "sport", "athlete",
    "games", "summer", "winter", "competitor", "nation", "venue", "event", "events",
    "swimming", "athletics", "gymnastics", "shooting", "cycling", "rowing", "boxing",
    "wrestling", "football", "tennis", "volleyball", "basketball", "skiing", "skating",
    "hockey", "archery", "fencing", "judo", "taekwondo", "weightlifting", "triathlon",
    "canoe", "kayak", "diving", "equestrian", "modern pentathlon", "sailing", "handball",
    "biathlon", "bobsled", "luge", "curling", "speed skating", "figure skating",
    "how many", "who won", "which event", "summarise", "summarize",
}

def _is_olympic_question(q: str) -> bool:
    ql = q.lower()
    return any(kw in ql for kw in _OLYMPIC_KEYWORDS)


def classify_question(q: str) -> str:
    """Heuristic question type classifier for live queries."""
    ql = q.lower()
    if any(w in ql for w in ("summarize", "summarise", "summary", "describe", "overview", "tell me about", "give me a")):
        return "summary"
    if any(w in ql for w in ("how many", "count", "total number", "number of events")):
        return "aggregation"
    if any(w in ql for w in ("most competitors", "most nations", "fewest", "highest", "lowest", "which event had the most", "which event had the fewest")):
        return "superlative"
    if any(w in ql for w in ("before", "after", "previous", "next", "immediately", "preceding", "prior")):
        return "temporal"
    if "held at" in ql or re.search(r'on \d{1,2}[\s\-–]+\w+', ql):
        return "multi_hop"
    return "lookup"


def _load(filename: str) -> dict:
    path = RESULTS_DIR / filename
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/results")
def get_all_results():
    return {
        "graphrag": _load("graphrag_results.json"),
        "agentic":  _load("agentic_results.json"),
        "rag":      _load("rag_results.json"),
    }


@app.get("/api/results/{pipeline}")
def get_pipeline_results(pipeline: str):
    mapping = {
        "graphrag": "graphrag_results.json",
        "agentic":  "agentic_results.json",
        "rag":      "rag_results.json",
    }
    if pipeline not in mapping:
        raise HTTPException(404, f"Unknown pipeline: {pipeline}")
    return _load(mapping[pipeline])


class QueryRequest(BaseModel):
    question: str
    pipeline: str = "graphrag"


@app.post("/api/query")
def run_live_query(req: QueryRequest):
    t0 = time.time()

    if not _is_olympic_question(req.question):
        return {
            "answer":          "I can only answer questions about Olympic events, sports, athletes, and results. Please ask something relevant to the Olympic dataset.",
            "graph_path":      "",
            "agent_steps":     [],
            "tokens":          0,
            "elapsed":         0,
            "pipeline":        req.pipeline,
            "qtype":           "off_topic",
            "doc_snippets":    [],
            "used_fallback":   False,
            "traversed_ids":   [],
            "traversed_nodes": [],
            "answer_node_ids": [],
        }

    qtype = classify_question(req.question)

    try:
        pipeline = get_pipeline(req.pipeline)
        result   = pipeline.answer(req.question, qid="live", qtype=qtype, gold=[])
    except Exception as e:
        raise HTTPException(500, detail=str(e))

    # Extract structured agentic steps before normalising graph_path
    agent_steps: list = []
    if req.pipeline == "agentic":
        raw_chain = result.get("evidence_chain") or []
        if isinstance(raw_chain, list):
            agent_steps = raw_chain  # list of "tool: input → output" strings

    # Normalise graph_path to a plain string
    raw_path = (
        result.get("graph_path")
        or result.get("evidence_chain")
        or result.get("evidence", "")
        or ""
    )
    if isinstance(raw_path, list):
        graph_path = " → ".join(str(s) for s in raw_path)
    else:
        graph_path = str(raw_path) if raw_path else ""

    # Retrieve doc titles for RAG
    doc_snippets = []
    if req.pipeline == "rag":
        try:
            from pipelines.shared import Corpus
            corpus = Corpus()
            for doc_id in (result.get("retrieved_doc_ids") or [])[:5]:
                title = corpus.get_title(doc_id)
                score_text = ""
                doc_snippets.append({"title": title, "doc_id": doc_id, "score": score_text})
        except Exception:
            pass

    # Return all retrieved IDs (no artificial cap) — visualization handles any length
    traversed_ids   = result.get("retrieved_doc_ids") or []
    answer_node_ids = result.get("answer_node_ids") or (traversed_ids[-1:] if traversed_ids else [])

    # Include full node data for traversed events so frontend can inject
    # them into Cytoscape even if they weren't in the pre-sampled graph
    traversed_nodes: list = []
    if req.pipeline == "graphrag":
        try:
            gp = _pipelines.get("graphrag")
            if gp and gp.events:
                ev_by_vid = {ev["vid"]: ev for ev in gp.events}
                for vid in traversed_ids:
                    ev = ev_by_vid.get(vid)
                    if ev:
                        traversed_nodes.append({
                            "id":          ev["vid"],
                            "label":       (ev.get("event_name") or ev["vid"])[:55],
                            "sport":       ev.get("sport", ""),
                            "year":        ev.get("year", 0),
                            "season":      ev.get("season", "Summer"),
                            "venue":       ev.get("venue_name", ""),
                            "competitors": ev.get("competitors") or 0,
                            "nations":     ev.get("nations") or 0,
                            "date_str":    ev.get("date_str", ""),
                        })
        except Exception:
            pass

    return {
        "answer":          result.get("predicted_answer", ""),
        "graph_path":      graph_path,
        "agent_steps":     agent_steps,
        "tokens":          result.get("tokens_used", 0),
        "elapsed":         round(time.time() - t0, 2),
        "pipeline":        req.pipeline,
        "qtype":           qtype,
        "doc_snippets":    doc_snippets,
        "used_fallback":   result.get("used_fallback", False),
        "traversed_ids":   traversed_ids,
        "traversed_nodes": traversed_nodes,
        "answer_node_ids": answer_node_ids,
        "query_trace":     result.get("query_trace") or [],
    }


def _build_edges(nodes: list) -> list:
    """Build edges: PREV_EDITION (same sport, consecutive years) + SAME_OLYMPICS (same year+season)."""
    import random as _random
    from collections import defaultdict
    rng = _random.Random(42)

    by_sport: dict = defaultdict(list)
    by_games: dict = defaultdict(list)
    for n in nodes:
        by_sport[(n["sport"], n["season"])].append(n)
        by_games[(n["year"], n["season"])].append(n)

    seen: set = set()
    edges: list = []

    def _add(a_id, b_id, etype):
        key = (min(a_id, b_id), max(a_id, b_id))
        if key not in seen:
            seen.add(key)
            edges.append({"source": a_id, "target": b_id, "type": etype})

    # PREV_EDITION: same sport+season, gap ≤ 8 years
    for group in by_sport.values():
        sg = sorted(group, key=lambda n: n["year"])
        for i in range(len(sg) - 1):
            a, b = sg[i], sg[i + 1]
            if 0 < b["year"] - a["year"] <= 8:
                _add(b["id"], a["id"], "PREV_EDITION")

    # SAME_OLYMPICS: random subset of pairs within same Games (adds density)
    for group in by_games.values():
        if len(group) < 2:
            continue
        pairs = [(group[i]["id"], group[j]["id"])
                 for i in range(len(group))
                 for j in range(i + 1, len(group))]
        k = min(len(group) * 3, len(pairs))
        for a_id, b_id in rng.sample(pairs, k):
            _add(a_id, b_id, "SAME_OLYMPICS")

    return edges


def _result_vids() -> set:
    """Collect all retrieved_doc_ids that appear in benchmark results."""
    seen: set = set()
    for pl in ("graphrag", "agentic", "rag"):
        data = _load(f"{pl}_results.json")
        for r in data.get("results", []):
            for did in (r.get("retrieved_doc_ids") or []):
                seen.add(did)
    return seen


def _graph_from_events(events: list, limit: int) -> dict:
    """Build graph payload from TigerGraph event objects.

    Always include events that appear in benchmark results so traversal
    highlighting works. Fill remaining slots with a sport-balanced sample.
    """
    from collections import defaultdict

    # Events that appear in results — always include these
    priority_vids = _result_vids()
    ev_by_vid = {ev["vid"]: ev for ev in events}

    priority = [ev_by_vid[v] for v in priority_vids if v in ev_by_vid]
    rest     = [ev for ev in events if ev["vid"] not in priority_vids]

    # Sample the rest by sport for visual variety
    by_sport: dict = defaultdict(list)
    for ev in rest:
        by_sport[ev.get("sport", "Other")].append(ev)
    slots = max(0, limit - len(priority))
    per_sport = max(2, slots // max(1, len(by_sport)))
    filler: list = []
    for evs in by_sport.values():
        filler.extend(evs[:per_sport])

    sampled = (priority + filler)[:limit]

    nodes = [{
        "id":          ev["vid"],
        "label":       (ev.get("event_name") or ev["vid"])[:55],
        "sport":       ev.get("sport", ""),
        "year":        ev.get("year", 0),
        "season":      ev.get("season", "Summer"),
        "venue":       ev.get("venue_name", ""),
        "competitors": ev.get("competitors") or 0,
        "nations":     ev.get("nations") or 0,
        "date_str":    ev.get("date_str", ""),
    } for ev in sampled]
    return {"nodes": nodes, "edges": _build_edges(nodes)}


def _graph_from_corpus(limit: int) -> dict:
    """Fallback: build graph from embedded doc titles + results data."""
    titles_path = ROOT / "embeddings" / "doc_titles.json"
    ids_path    = ROOT / "embeddings" / "doc_ids.json"
    if not titles_path.exists() or not ids_path.exists():
        return {"nodes": [], "edges": []}

    with open(titles_path, encoding="utf-8") as f:
        titles: dict = json.load(f)   # {doc_id: title}
    with open(ids_path, encoding="utf-8") as f:
        all_ids: list = json.load(f)

    # Pull doc_ids seen in benchmark results (for relevance)
    seen_ids: set = set()
    for pl in ("graphrag", "agentic", "rag"):
        data = _load(f"{pl}_results.json")
        for r in data.get("results", []):
            for did in (r.get("retrieved_doc_ids") or []):
                seen_ids.add(did)

    # Prioritise seen IDs, then fill with others up to limit
    pool = list(seen_ids) + [d for d in all_ids if d not in seen_ids]
    pool = pool[:limit]

    year_re   = re.compile(r'\b(19|20)\d{2}\b')
    season_re = re.compile(r'\b(Summer|Winter)\b')
    sport_re  = re.compile(r'^([A-Za-z][A-Za-z\s/]+?)\s+at\s+the\s+\d')

    nodes = []
    for doc_id in pool:
        title = titles.get(doc_id, "")
        if not title:
            continue
        ym = year_re.search(title)
        sm = season_re.search(title)
        pm = sport_re.match(title)
        nodes.append({
            "id":          doc_id,
            "label":       title[:55],
            "sport":       pm.group(1).strip() if pm else "Other",
            "year":        int(ym.group()) if ym else 0,
            "season":      sm.group() if sm else "Summer",
            "venue":       "",
            "competitors": 0,
            "nations":     0,
            "date_str":    "",
        })

    return {"nodes": nodes, "edges": _build_edges(nodes)}


@app.get("/api/graph")
def get_graph_overview(limit: int = 160):
    """Return a sampled OlympicEvent subgraph for the interactive visualization."""
    try:
        pipeline = get_pipeline("graphrag")
        if pipeline.events:
            return _graph_from_events(pipeline.events, limit)
    except Exception:
        pass
    # TigerGraph unavailable — fall back to corpus-derived graph
    return _graph_from_corpus(limit)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "pipelines_loaded": list(_pipelines.keys()),
        "results_available": {
            k: (RESULTS_DIR / f"{k}_results.json").exists()
            for k in ("graphrag", "agentic", "rag")
        },
    }
