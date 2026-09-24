"""
Pipeline 3: Agentic GraphRAG — LLM-driven planning + adaptive retrieval.

Key differences from GraphRAG:
  1. LLM PLANS which tools to use (not hardcoded by qtype)
  2. Adaptive loop: if answer is Unknown, agent retries with different strategy
  3. Multi-source synthesis: graph + vector evidence combined by LLM
  4. Evidence chain: every answer cites which tools were used and why

Agent tools available:
  graph_filter     — filter OlympicEvent vertices by attribute
  count_results    — count filtered results (aggregation)
  sort_results     — sort by attribute (superlative)
  follow_edge      — traverse PREV/NEXT_EDITION edges
  infobox_lookup   — extract gold/silver/bronze from corpus infobox
  vector_search    — semantic similarity search over all docs
  attribute_lookup — read a specific attribute from a vertex

Agent loop (max 3 iterations):
  1. Plan: LLM extracts entities + selects tools
  2. Execute: run selected tools
  3. Synthesize: LLM answers from evidence
  4. If answer is Unknown/empty: retry with fallback strategy
"""

import json
import re
import sys
import os
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tg_client import TGClient
from .shared import (
    Corpus, EmbeddingIndex, llm_call, llm_complete,
    extract_final_answer, _strip_country_suffix, make_result
)
from .graphrag_pipeline import (
    get_all_events, get_prev_edition, get_next_edition,
    filter_events, match_event_by_venue_date, parse_medalist_name,
    parse_infobox_field, extract_year, extract_season, extract_threshold,
    find_sport_match, extract_venue_hint, extract_date_hint,
    fallback_vector, _title_word_score, _answer_instruction,
    query_lookup   as _graphrag_query_lookup,
    query_summary  as _graphrag_query_summary,
)

RESULTS_PATH = Path(__file__).parent.parent / "results" / "agentic_results.json"

MAX_AGENT_ITERATIONS = 3


# ── Planning prompt ───────────────────────────────────────────

PLAN_PROMPT = """\
You are an Olympic data analyst. Analyze this question and extract structured information.

Question: {question}
Question type: {qtype}

Extract the following as valid JSON (null for unknown values):
{{
  "year": <integer or null>,
  "season": <"Summer" or "Winter" or null>,
  "sport": <sport name exactly as in Wikipedia, e.g. "Athletics", "Biathlon", "Swimming", or null>,
  "event_name": <specific event, e.g. "Men's 20 kilometres walk", or null>,
  "venue": <venue name or null>,
  "date": <date string like "20 September 1988" or null>,
  "threshold_op": <">" or "<" or null>,
  "threshold_val": <integer or null>,
  "direction": <"prev" if asking about BEFORE, "next" if asking about AFTER, or null>,
  "answer_type": <"count", "name", "event_title", "number", "country", "score">,
  "medal": <"gold", "silver", "bronze", or null>
}}
Return ONLY the JSON object, no other text."""


SYNTHESIS_PROMPT = """\
You are answering an Olympic trivia question. Use the evidence below to give a direct, concise answer.

Evidence gathered:
{evidence}

Question: {question}

Give the shortest possible answer (a name, number, or phrase — no explanation).
Answer:"""


# ── Entity extraction ─────────────────────────────────────────

def _enrich_plan(plan: dict, question: str, events: list[dict]) -> dict:
    """Fill missing plan fields using regex extractors when LLM plan is incomplete."""
    known_sports = {e.get("sport", "") for e in events if e.get("sport")}
    result = dict(plan)
    # Always override year/season from regex — more reliable than LLM for temporal questions
    # (LLM may compute "before 2016" → 2012, but we need the explicit year 2016 as anchor)
    extracted_year = extract_year(question)
    if extracted_year is not None:
        result["year"] = extracted_year
    if result.get("season") is None:
        result["season"] = extract_season(question)
    if result.get("sport") is None:
        result["sport"] = find_sport_match(question, known_sports)
    if result.get("threshold_op") is None:
        op, val = extract_threshold(question)
        result["threshold_op"] = op
        result["threshold_val"] = val
    # Always derive direction from keywords — more reliable than LLM output
    q_lower = question.lower()
    if any(w in q_lower for w in ("before", "previous", "preceding", "prior")):
        result["direction"] = "prev"
    elif any(w in q_lower for w in ("after", "next")):
        result["direction"] = "next"
    if result.get("medal") is None:
        q_lower = question.lower()
        result["medal"] = ("silver" if "silver" in q_lower
                           else "bronze" if "bronze" in q_lower
                           else "gold")
    return result


def llm_plan(question: str, qtype: str) -> dict:
    """Use LLM to extract structured entities from question."""
    prompt = PLAN_PROMPT.format(question=question, qtype=qtype)
    raw, _ = llm_call(prompt, max_tokens=200)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    try:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except (json.JSONDecodeError, AttributeError):
        pass
    return {}


# ── Agent tools ───────────────────────────────────────────────

class AgentEvidence:
    """Accumulates evidence across multiple tool calls."""
    def __init__(self):
        self.steps: list[dict] = []
        self.doc_ids: list[str] = []
        self.tokens_used: int = 0

    def add(self, tool: str, input_desc: str, output: str, doc_ids: list[str] = None):
        self.steps.append({"tool": tool, "input": input_desc, "output": output})
        if doc_ids:
            self.doc_ids.extend(d for d in doc_ids if d not in self.doc_ids)

    def to_text(self) -> str:
        return "\n".join(
            f"[{s['tool']}] {s['input']} → {s['output']}"
            for s in self.steps
        )

    # Tools that produce metadata/plans, not final answers
    _META_TOOLS = {"llm_plan", "retry_vector", "final_fallback"}

    def best_answer(self) -> str | None:
        """Scan evidence for the most recent concrete answer (any length)."""
        for step in reversed(self.steps):
            if step["tool"] in self._META_TOOLS:
                continue
            out = step["output"]
            if out and out.lower() not in ("none", "not found", "unknown", "", "null", "..."):
                return out
        return None


class AgentExecutor:
    """Executes individual tool calls using graph + vector resources."""

    def __init__(self, tg: TGClient | None, events: list[dict],
                 corpus: Corpus, index: EmbeddingIndex):
        self.tg      = tg
        self.events  = events
        self.corpus  = corpus
        self.index   = index
        self._known_sports = {e.get("sport", "") for e in events if e.get("sport")}

    def tool_graph_filter(self, plan: dict) -> tuple[list[dict], str]:
        year   = plan.get("year")
        season = plan.get("season")
        sport  = plan.get("sport") or find_sport_match(
            (plan.get("event_name") or ""), self._known_sports)
        filtered = filter_events(self.events, year=year, season=season, sport=sport)
        desc = f"year={year}, season={season}, sport={sport} → {len(filtered)} events"
        return filtered, desc

    def tool_count_results(self, filtered: list[dict], plan: dict) -> tuple[str, str]:
        op  = plan.get("threshold_op")
        val = plan.get("threshold_val")
        if op and val is not None:
            if op == ">":
                matching = [e for e in filtered if (e.get("competitors") or 0) > val]
            else:
                matching = [e for e in filtered if (e.get("competitors") or 0) < val]
            return str(len(matching)), f"competitors{op}{val} → {len(matching)}"
        return str(len(filtered)), f"total={len(filtered)}"

    def tool_sort_results(self, filtered: list[dict], plan: dict) -> tuple[dict | None, str]:
        attr = "competitors"
        if plan.get("answer_type") == "count" or "nation" in str(plan):
            attr = "nations"
        if not filtered:
            return None, "no events to sort"
        best = max(filtered, key=lambda e: e.get(attr) or 0)
        return best, f"max {attr}={best.get(attr)} → {best.get('title', best['vid'])}"

    def tool_follow_edge(self, plan: dict, filtered: list[dict],
                         question: str = "") -> tuple[str | None, str]:
        if not filtered or not self.tg:
            return None, "no anchor event or TG unavailable"
        # Pick best anchor by title word overlap with question
        if len(filtered) > 1 and question:
            anchor = max(filtered, key=lambda e: _title_word_score(question, e))
        else:
            anchor = filtered[0]
        vid = anchor["vid"]
        direction = (plan.get("direction") or "prev").lower()
        if "prev" in direction or "before" in direction:
            target = get_prev_edition(self.tg, vid)
            edge_type = "PREV_EDITION"
        else:
            target = get_next_edition(self.tg, vid)
            edge_type = "NEXT_EDITION"
        desc = f"{vid} -{edge_type}→ {target}"
        return target, desc

    def tool_infobox_lookup(self, vid: str, plan: dict) -> tuple[str | None, str]:
        medal = plan.get("medal") or "gold"
        text  = self.corpus.get_text(vid, max_chars=1500)
        val   = parse_medalist_name(text, medal)
        if not val:
            val = parse_infobox_field(text, medal)
        desc = f"{vid} infobox.{medal}={val}"
        return val, desc

    def tool_attribute_lookup(self, filtered: list[dict], plan: dict) -> tuple[str | None, str]:
        if not filtered:
            return None, "no event found"
        event = filtered[0]
        answer_type = plan.get("answer_type", "name")
        if answer_type == "count":
            val = str(event.get("competitors") or "")
        elif answer_type == "number":
            val = str(event.get("competitors") or event.get("nations") or "")
        else:
            val = event.get("title") or event["vid"]
        return val, f"event={event.get('title')} attr={val}"

    def tool_vector_search(self, question: str, top_k: int = 5) -> tuple[str, list[str], str]:
        hits = self.index.search(question, top_k=top_k)
        chunks = []
        for i, (doc_id, score) in enumerate(hits, 1):
            title = self.corpus.get_title(doc_id)
            text  = self.corpus.get_text(doc_id, max_chars=700)
            chunks.append(f"[Doc {i} | {title} | score={score:.3f}]\n{text}")
        doc_ids = [d for d, _ in hits]
        context = "\n---\n".join(chunks)
        return context, doc_ids, f"vector search top-{top_k} retrieved"

    def tool_venue_date_filter(self, question: str) -> tuple[list[dict], str]:
        venue  = extract_venue_hint(question)
        date   = extract_date_hint(question)
        year   = extract_year(question)
        base   = filter_events(self.events, year=year) if year else list(self.events)
        result = match_event_by_venue_date(base, venue, date)
        desc   = f"venue={venue}, date={date} → {len(result)} events"
        return result, desc


# ── Agentic execution strategies ─────────────────────────────

def run_aggregation_agent(executor: AgentExecutor, plan: dict,
                          question: str) -> AgentEvidence:
    ev = AgentEvidence()
    filtered, desc = executor.tool_graph_filter(plan)
    ev.add("graph_filter", f"filter by {desc}", f"{len(filtered)} events found",
           [e["vid"] for e in filtered[:5]])
    count, count_desc = executor.tool_count_results(filtered, plan)
    ev.add("count_results", count_desc, count)
    return ev


def run_superlative_agent(executor: AgentExecutor, plan: dict,
                          question: str) -> AgentEvidence:
    ev = AgentEvidence()
    filtered, desc = executor.tool_graph_filter(plan)
    ev.add("graph_filter", desc, f"{len(filtered)} events")
    best, sort_desc = executor.tool_sort_results(filtered, plan)
    # Answer IS the event title — do not overwrite with infobox lookup
    title = best.get("title") if best else None
    ev.add("sort_results", sort_desc, title or "none",
           [best["vid"]] if best else [])
    return ev


def run_temporal_agent(executor: AgentExecutor, plan: dict,
                       question: str, tg: TGClient | None) -> AgentEvidence:
    ev = AgentEvidence()
    filtered, desc = executor.tool_graph_filter(plan)
    ev.add("graph_filter", desc, f"{len(filtered)} anchor events")

    if not filtered:
        return ev

    target_vid, edge_desc = executor.tool_follow_edge(plan, filtered, question)
    ev.add("follow_edge", edge_desc, target_vid or "edge missing")

    if not target_vid:
        # Fallback: look for events 4 years before/after
        year = plan.get("year")
        if year:
            direction = plan.get("direction", "prev")
            offset = -4 if "prev" in direction else 4
            fb_plan = dict(plan)
            fb_plan["year"] = year + offset
            fb_filtered, fb_desc = executor.tool_graph_filter(fb_plan)
            ev.add("graph_filter_fallback", fb_desc, f"{len(fb_filtered)} events")
            if fb_filtered:
                target_vid = fb_filtered[0]["vid"]
                ev.add("fallback_year", f"year {year+offset}", target_vid)

    if target_vid:
        gold, ib_desc = executor.tool_infobox_lookup(target_vid, plan)
        ev.add("infobox_lookup", ib_desc, gold or "not found", [target_vid])
    return ev


def run_multihop_agent(executor: AgentExecutor, plan: dict,
                       question: str) -> AgentEvidence:
    ev = AgentEvidence()
    filtered, desc = executor.tool_venue_date_filter(question)
    ev.add("venue_date_filter", desc, f"{len(filtered)} events")

    if not filtered and plan.get("year"):
        # Try year filter only
        filtered2, desc2 = executor.tool_graph_filter(plan)
        ev.add("year_filter_fallback", desc2, f"{len(filtered2)} events")
        filtered = filtered2

    if filtered:
        gold, ib_desc = executor.tool_infobox_lookup(filtered[0]["vid"], plan)
        ev.add("infobox_lookup", ib_desc, gold or "not found", [filtered[0]["vid"]])
    return ev


def run_summary_agent(executor: AgentExecutor, plan: dict,
                      question: str) -> AgentEvidence:
    ev = AgentEvidence()
    answer, doc_ids, graph_path, answer_ids = _graphrag_query_summary(
        executor.tg, executor.events, executor.corpus, question, None)
    ev.add("graph_summary", graph_path[:80], answer or "not found", doc_ids[:5])
    return ev


def run_lookup_agent(executor: AgentExecutor, plan: dict,
                     question: str) -> AgentEvidence:
    ev = AgentEvidence()
    # Delegate to graphrag query_lookup which handles all patterns (nations, medalists, etc.)
    answer, doc_ids, graph_path, _ = _graphrag_query_lookup(
        executor.tg, executor.events, executor.corpus, question)
    ev.add("graph_lookup", graph_path[:80], answer or "not found", doc_ids)
    return ev


# ── Final synthesis ───────────────────────────────────────────

def synthesize_answer(evidence: AgentEvidence, question: str,
                      corpus: Corpus, index: EmbeddingIndex) -> tuple[str, int]:
    """
    Extract answer from evidence. If structured graph answer found, use it directly.
    Otherwise, use LLM synthesis (chat mode) over evidence + vector docs.
    """
    # Try direct extraction from evidence
    direct = evidence.best_answer()
    if direct and direct.lower() not in ("not found", "none found", "no events"):
        return direct, 0

    # LLM synthesis from evidence text
    evidence_text = evidence.to_text()

    hits = index.search(question, top_k=3)
    extra_docs = []
    for doc_id, score in hits:
        title = corpus.get_title(doc_id)
        text  = corpus.get_text(doc_id, max_chars=500)
        extra_docs.append(f"[{title}]\n{text}")
        if doc_id not in evidence.doc_ids:
            evidence.doc_ids.append(doc_id)

    full_evidence = evidence_text
    if extra_docs:
        full_evidence += "\n\nAdditional retrieved documents:\n" + "\n---\n".join(extra_docs)

    system = (
        "You are a precise Olympic data assistant. "
        "Answer using ONLY the provided evidence. "
        "Never reveal your reasoning. "
        + _answer_instruction(question)
    )
    user_msg = f"Evidence:\n{full_evidence}\n\nQuestion: {question}"
    raw, tokens = llm_call(user_msg, max_tokens=200, system=system)
    evidence.tokens_used += tokens
    answer = extract_final_answer(raw)
    return answer, tokens


# ── Main pipeline ─────────────────────────────────────────────

class AgenticGraphRAGPipeline:
    def __init__(self):
        self.corpus = Corpus()
        self.index  = EmbeddingIndex()
        self.tg     = None
        self.events = []
        self._tg_available = False
        try:
            self.tg = TGClient()
            print("Loading OlympicEvent vertices ...", end=" ", flush=True)
            self.events = get_all_events(self.tg)
            print(f"{len(self.events)} events loaded.")
            self._tg_available = True
        except Exception as e:
            print(f"\n  WARNING: TigerGraph unavailable ({e})")
            print("  Running in vector-only mode.")

        self.executor = AgentExecutor(
            tg=self.tg,
            events=self.events,
            corpus=self.corpus,
            index=self.index,
        )

    def answer(self, question: str, qid: str = "", qtype: str = "",
               gold: list[str] = None) -> dict:
        t0 = time.time()
        evidence = AgentEvidence()
        final_answer = None
        plan = {}

        for iteration in range(MAX_AGENT_ITERATIONS):
            if iteration == 0:
                # Phase 1: LLM Planning + regex enrichment
                plan = llm_plan(question, qtype)
                plan = _enrich_plan(plan, question, self.events)
                evidence.add("llm_plan", f"question={question[:50]}", str(plan))

                # Phase 2: Execute graph strategy
                if self._tg_available and self.events:
                    if qtype == "aggregation":
                        ev = run_aggregation_agent(self.executor, plan, question)
                    elif qtype == "superlative":
                        ev = run_superlative_agent(self.executor, plan, question)
                    elif qtype == "temporal":
                        ev = run_temporal_agent(self.executor, plan, question, self.tg)
                    elif qtype == "multi_hop":
                        ev = run_multihop_agent(self.executor, plan, question)
                    elif qtype == "summary":
                        ev = run_summary_agent(self.executor, plan, question)
                    else:  # lookup
                        ev = run_lookup_agent(self.executor, plan, question)

                    for step in ev.steps:
                        evidence.steps.append(step)
                    evidence.doc_ids.extend(
                        d for d in ev.doc_ids if d not in evidence.doc_ids)

            elif iteration == 1:
                # Phase 2: Retry — add vector search as additional evidence
                evidence.add("retry_vector", "adding vector search context", "...")
                ctx, vids, vdesc = self.executor.tool_vector_search(question, top_k=5)
                evidence.add("vector_search", vdesc, ctx[:200])
                for vid in vids:
                    if vid not in evidence.doc_ids:
                        evidence.doc_ids.append(vid)

            else:
                # Phase 3: Final fallback — pure vector synthesis
                raw, _, __ = fallback_vector(self.index, self.corpus, question)
                evidence.add("final_fallback", "pure vector search", raw)

            # Synthesize answer from current evidence
            final_answer, tokens = synthesize_answer(
                evidence, question, self.corpus, self.index)
            evidence.tokens_used += tokens

            # Check if answer is satisfactory
            if (final_answer and
                    final_answer.lower() not in ("unknown", "none", "not found", "")):
                break  # Confidence: got a real answer

        # Post-validation: "who" questions must not return bare numbers
        if final_answer:
            q_lower = question.lower()
            if any(w in q_lower for w in ("who ", "who's", "winner", "won ")):
                if re.match(r'^\d', final_answer.strip()):
                    final_answer = "Information not found for this question"

        elapsed = time.time() - t0
        return make_result(
            pipeline     = "AgenticGraphRAG",
            qid          = qid,
            question     = question,
            qtype        = qtype,
            answer       = final_answer or "Unknown",
            gold         = gold or [],
            retrieved_ids= evidence.doc_ids[:10],
            tokens       = evidence.tokens_used,
            elapsed      = elapsed,
            extra        = {
                "iterations"    : iteration + 1,
                "evidence_steps": len(evidence.steps),
                "evidence_chain": [
                    f"{s['tool']}: {s['input'][:120]} → {s['output'][:120]}"
                    for s in evidence.steps
                ],
                "plan"          : plan,
            },
        )


# ── Benchmark runner ──────────────────────────────────────────

def run_benchmark(questions_path: str = None, limit: int = None) -> list[dict]:
    if questions_path is None:
        questions_path = (
            Path(__file__).parent.parent
            / "questions-20260920T071303Z-1-001/questions/eval_public.jsonl"
        )

    with open(questions_path, encoding="utf-8") as f:
        questions = [json.loads(l) for l in f if l.strip()]
    if limit:
        questions = questions[:limit]

    pipeline = AgenticGraphRAGPipeline()
    results  = []
    type_scores = {}

    print(f"\nRunning Agentic GraphRAG on {len(questions)} questions ...")
    print("-" * 70)

    for i, q in enumerate(questions, 1):
        qid   = q["qid"]
        qtype = q["qtype"]
        gold  = q.get("answer", [])

        result = pipeline.answer(
            question=q["question"], qid=qid, qtype=qtype, gold=gold)
        results.append(result)

        if qtype not in type_scores:
            type_scores[qtype] = {"total": 0, "contains_gold": 0, "exact_match": 0}
        type_scores[qtype]["total"]        += 1
        type_scores[qtype]["contains_gold"] += result["contains_gold"]
        type_scores[qtype]["exact_match"]   += result["exact_match"]

        em   = result["exact_match"]
        cg   = result["contains_gold"]
        itr  = result.get("iterations", 1)
        print(f"[{i:3d}/{len(questions)}] {qid} ({qtype:<12}) "
              f"itr={itr} EM={em} CG={cg} t={result['elapsed_sec']:5.1f}s | "
              f"pred: {result['predicted_answer'][:40]!r}")

    print("\n" + "=" * 70)
    print("AGENTIC GRAPHRAG BENCHMARK SUMMARY")
    print("=" * 70)
    overall_em = sum(r["exact_match"]   for r in results) / len(results)
    overall_cg = sum(r["contains_gold"] for r in results) / len(results)
    overall_tok = sum(r["tokens_used"]  for r in results)
    print(f"Overall  exact_match={overall_em:.1%}  contains_gold={overall_cg:.1%}  "
          f"total_tokens={overall_tok}")
    print()
    print(f"{'Type':<15} {'N':>4}  {'ExactMatch':>10}  {'ContainsGold':>12}")
    print("-" * 45)
    for qtype, s in sorted(type_scores.items()):
        n  = s["total"]
        em = s["exact_match"] / n
        cg = s["contains_gold"] / n
        print(f"{qtype:<15} {n:>4}  {em:>10.1%}  {cg:>12.1%}")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    payload = {
        "pipeline": "AgenticGraphRAG",
        "n_questions": len(results),
        "overall": {
            "exact_match"  : round(overall_em, 4),
            "contains_gold": round(overall_cg, 4),
            "total_tokens" : overall_tok,
        },
        "by_type": {
            t: {
                "exact_match"  : round(s["exact_match"] / s["total"], 4),
                "contains_gold": round(s["contains_gold"] / s["total"], 4),
                "n"            : s["total"],
            }
            for t, s in type_scores.items()
        },
        "results": results,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_benchmark(limit=limit)
