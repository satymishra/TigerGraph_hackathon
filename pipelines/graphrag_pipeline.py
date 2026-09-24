"""
Pipeline 2: GraphRAG — structured entity extraction + TigerGraph traversal.

Approach:
  - Query TigerGraph for structured graph data (events, edges)
  - Parse gold medalist names from corpus infoboxes
  - Fallback to vector search when graph finds no match
  - LLM used only for final answer synthesis (minimal)

Strategy by question type:
  aggregation  → filter events(sport, year, season) → count by competitors threshold
  superlative  → filter events(sport, season) → sort by competitors/nations
  temporal     → find event(year, sport) → follow PREV_EDITION edge → gold from infobox
  multi_hop    → find event(venue + date) → gold from infobox
  lookup       → find event(year + sport/event_name) → gold from infobox
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
    Corpus, EmbeddingIndex, llm_call, llm_complete, extract_final_answer,
    _strip_country_suffix, make_result
)

RESULTS_PATH = Path(__file__).parent.parent / "results" / "graphrag_results.json"


# ── Infobox parsing ───────────────────────────────────────────

def parse_infobox_field(text: str, field: str) -> str | None:
    """Extract a field from Wikipedia infobox text.

    Corpus uses preprocessed format: '  field: value' (colon-space separated).
    Falls back to raw wiki format: '|field=value'.
    """
    def _clean(val):
        val = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]]+)\]\]', r'\1', val)
        val = re.sub(r'\{\{.*?\}\}', '', val)
        val = re.sub(r"'{2,}", '', val)
        return val.strip() or None

    # Primary: preprocessed corpus format "  field: value"
    m = re.search(rf'^\s*{re.escape(field)}\s*:\s*([^\n]+)',
                  text, re.IGNORECASE | re.MULTILINE)
    if m:
        return _clean(m.group(1))

    # Fallback: raw wiki format "|field=value"
    m = re.search(rf'\|\s*{re.escape(field)}\s*=\s*([^\|\n\]]+)',
                  text, re.IGNORECASE)
    if m:
        return _clean(m.group(1))

    return None


def parse_medalist_name(text: str, medal: str) -> str | None:
    """Extract gold/silver/bronze medalist from infobox, clean up."""
    # Try multiple Wikipedia field name variants
    variants = [
        medal,
        f"{medal} medalist",
        f"{medal} medallist",
        f"{medal} athlete",
        f"{medal}_medalist",
        f"{medal}_medallist",
        f"{medal}_athlete",
    ]
    raw = None
    for v in variants:
        raw = parse_infobox_field(text, v)
        if raw:
            break
    if not raw:
        return None
    raw = _strip_country_suffix(raw)
    # Strip "flagathlete" template leftovers and parenthetical suffixes
    raw = re.sub(r'\s*\(.*?\)', '', raw).strip()
    return raw or None


def _extract_name_from_text(text: str) -> str | None:
    """Last-resort: pull the first capitalized full name from document text."""
    if not text:
        return None
    m = re.search(r'\b([A-ZÁÉÍÓÚÀÈÜÄ][a-záéíóúàèüä]+(?:\s+[A-ZÁÉÍÓÚÀÈÜÄ][a-záéíóúàèüä]+){1,3})\b', text)
    return m.group(1) if m else None


# ── Entity extraction from question ──────────────────────────

def extract_year(question: str) -> int | None:
    m = re.search(r'\b(1\d{3}|20[0-2]\d)\b', question)
    return int(m.group()) if m else None


def extract_season(question: str) -> str | None:
    q = question.lower()
    if "winter" in q:
        return "Winter"
    if "summer" in q:
        return "Summer"
    return None


def extract_threshold(question: str) -> tuple[str, int] | tuple[None, None]:
    """Return (operator, value) for numeric comparisons, e.g. ('>', 73)."""
    m = re.search(r'(more than|greater than|at least|less than|fewer than|exactly)\s+(\d+)', question, re.IGNORECASE)
    if m:
        op_word = m.group(1).lower()
        val = int(m.group(2))
        op = ">" if "more" in op_word or "greater" in op_word or "least" in op_word else "<"
        return op, val
    m = re.search(r'>(\d+)', question)
    if m:
        return ">", int(m.group(1))
    return None, None


def find_sport_match(question: str, known_sports: set[str]) -> str | None:
    """Match question text against known sport names (longest match first)."""
    q = question.lower()
    for sport in sorted(known_sports, key=len, reverse=True):
        if sport.lower() in q:
            return sport
    return None


def extract_venue_hint(question: str) -> str | None:
    """Extract venue name hint from question.

    Multi_hop questions always follow 'held at VENUE on DATE' pattern.
    """
    # Primary: "held at VENUE on DATE" — captures any venue name
    m = re.search(r'held at (.+?)(?:\s+on\s+|\?|$)', question, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: "at VENUE" with known suffixes
    m = re.search(
        r'(?:at|in)\s+(?:the\s+)?([A-Z][a-zA-Z\s\-–\'\.]+?'
        r'(?:Stadium|Arena|Gymnasium|Park|Pool|Centre|Center|Hall|'
        r'Velodrome|Velopark|Pavilion|Barracks|Oval|Track|Ring|Field|Rink))',
        question)
    if m:
        return m.group(1).strip()
    return None


_MONTHS = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'

def extract_date_hint(question: str) -> str | None:
    """Extract a date string from question. Returns 'DD Month' for matching against date_str."""
    # Date range start FIRST: "11–19 August" or "3 to 4 August" → "11 August" / "3 August"
    m = re.search(rf'\b(\d{{1,2}})\s*(?:–|-|to)\s*\d{{1,2}}\s+({_MONTHS})', question, re.IGNORECASE)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    # DD Month [YYYY]  e.g. "6 August 2016", "11 August"
    m = re.search(rf'\b(\d{{1,2}})\s+({_MONTHS})', question, re.IGNORECASE)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    # Month DD[, YYYY]  e.g. "February 24, 2006", "August 10, 2008"
    m = re.search(rf'\b({_MONTHS})\s+(\d{{1,2}})\b', question, re.IGNORECASE)
    if m:
        return f"{m.group(2)} {m.group(1)}"
    return None


# ── TigerGraph REST++ queries ─────────────────────────────────

def get_all_events(tg: TGClient) -> list[dict]:
    """Fetch all OlympicEvent vertices from TigerGraph."""
    r = tg.get(f"/restpp/graph/{tg.graph}/vertices/OlympicEvent", params={"limit": 10000})
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"TG error: {data.get('message')}")
    return [{"vid": v["v_id"], **v["attributes"]} for v in data.get("results", [])]


def get_prev_edition(tg: TGClient, event_vid: str) -> str | None:
    """Follow PREV_EDITION edge from an OlympicEvent, return target vid."""
    r = tg.get(f"/restpp/graph/{tg.graph}/edges/OlympicEvent/{event_vid}/PREV_EDITION")
    data = r.json()
    if data.get("error") or not data.get("results"):
        return None
    return data["results"][0].get("to_id")


def get_next_edition(tg: TGClient, event_vid: str) -> str | None:
    """Follow NEXT_EDITION edge from an OlympicEvent, return target vid."""
    r = tg.get(f"/restpp/graph/{tg.graph}/edges/OlympicEvent/{event_vid}/NEXT_EDITION")
    data = r.json()
    if data.get("error") or not data.get("results"):
        return None
    return data["results"][0].get("to_id")


# ── Matching helpers ──────────────────────────────────────────

def filter_events(events: list[dict], year: int = None, season: str = None,
                  sport: str = None) -> list[dict]:
    """Filter event list by attributes."""
    result = events
    if year is not None:
        result = [e for e in result if e.get("year") == year]
    if season:
        result = [e for e in result if e.get("season", "").lower() == season.lower()]
    if sport:
        result = [e for e in result if sport.lower() in e.get("sport", "").lower()]
    return result


def match_event_by_title_fragment(events: list[dict], fragment: str) -> list[dict]:
    """Fuzzy title match: fragment words must appear in title."""
    words = [w for w in fragment.lower().split() if len(w) > 3]
    if not words:
        return []
    return [e for e in events if all(w in e.get("title", "").lower() for w in words)]


def match_event_by_venue_date(events: list[dict], venue_hint: str = None,
                               date_hint: str = None) -> list[dict]:
    """Find event by venue name and/or date string."""
    result = events
    if venue_hint:
        # Include all tokens (even single digits like "3") so "Arena 3" != "Arena 2"
        key_words = re.findall(r'\w+', venue_hint.lower())
        result = [e for e in result
                  if all(w in e.get("venue_name", "").lower() for w in key_words)]
    if date_hint:
        # Match day + month (year might differ in data)
        parts = date_hint.lower().split()
        result = [e for e in result
                  if all(p in e.get("date_str", "").lower() for p in parts[:2])]
    return result


# ── Query strategies by type ──────────────────────────────────

def query_aggregation(tg: TGClient, events: list[dict], corpus: Corpus,
                      question: str) -> tuple[str, list[str], str, list[str]]:
    """
    Returns (answer, all_doc_ids, graph_path, answer_ids).
    """
    year    = extract_year(question)
    season  = extract_season(question)
    op, thr = extract_threshold(question)

    known_sports = {e.get("sport", "") for e in events if e.get("sport")}
    sport = find_sport_match(question, known_sports)

    filtered = filter_events(events, year=year, season=season, sport=sport)

    graph_path = (f"OlympicEvent filter: year={year}, season={season}, sport={sport} "
                  f"→ {len(filtered)} events found")

    if not filtered:
        return None, [], graph_path, []

    all_ids = [e["vid"] for e in filtered]

    if op and thr is not None:
        if op == ">":
            matching = [e for e in filtered if (e.get("competitors") or 0) > thr]
        else:
            matching = [e for e in filtered if (e.get("competitors") or 0) < thr]
        answer = str(len(matching))
        graph_path += f" → competitors{op}{thr} → {len(matching)} events match"
        answer_ids = [e["vid"] for e in matching]
    else:
        answer = str(len(filtered))
        answer_ids = all_ids

    return answer, all_ids, graph_path, answer_ids


def query_superlative(tg: TGClient, events: list[dict], corpus: Corpus,
                      question: str) -> tuple[str, list[str], str, list[str]]:
    """Find event with most/fewest competitors or nations."""
    year   = extract_year(question)
    season = extract_season(question)
    known_sports = {e.get("sport", "") for e in events if e.get("sport")}
    sport  = find_sport_match(question, known_sports)

    filtered = filter_events(events, year=year, season=season, sport=sport)
    graph_path = f"Filter: year={year}, season={season}, sport={sport} → {len(filtered)} events"

    if not filtered:
        return None, [], graph_path, []

    q_lower = question.lower()
    if any(w in q_lower for w in ("most", "largest", "highest", "maximum", "greatest")):
        attr = "competitors" if "competitor" in q_lower else "nations"
        best = max(filtered, key=lambda e: e.get(attr) or 0)
    else:
        attr = "competitors" if "competitor" in q_lower else "nations"
        best = min(filtered, key=lambda e: e.get(attr) or 0)

    doc_text = corpus.get_text(best["vid"], max_chars=1500)
    gold = parse_medalist_name(doc_text, "gold")
    graph_path += f" → best event: {best.get('title', best['vid'])}"

    if "gold" in q_lower or "winner" in q_lower or "won" in q_lower:
        answer = gold or best.get("title", "Unknown")
    else:
        answer = best.get("title") or best["vid"]

    # Return all candidates so visualization shows the search space; best goes last
    all_ids = [e["vid"] for e in filtered if e["vid"] != best["vid"]] + [best["vid"]]
    return answer, all_ids, graph_path, [best["vid"]]


def query_temporal(tg: TGClient, events: list[dict], corpus: Corpus,
                   question: str) -> tuple[str, list[str], str]:
    """
    Temporal: find event at year X, follow PREV/NEXT edge, return gold.
    """
    year   = extract_year(question)
    season = extract_season(question)
    known_sports = {e.get("sport", "") for e in events if e.get("sport")}
    sport  = find_sport_match(question, known_sports)

    q_lower = question.lower()
    # Determine direction: "before", "previous", "preceding" → PREV; "after", "next" → NEXT
    direction = "PREV" if any(w in q_lower for w in ("before", "previous", "preceding", "prior")) else "NEXT"

    # Find the anchor event at 'year'
    anchor_candidates = filter_events(events, year=year, season=season, sport=sport)
    graph_path = f"Anchor: year={year}, season={season}, sport={sport} → {len(anchor_candidates)} candidates"

    if not anchor_candidates:
        return None, [], graph_path

    # If multiple, pick best title match using keywords from question
    anchor = anchor_candidates[0]
    if len(anchor_candidates) > 1:
        # Extract significant words from question (skip common words + sport name)
        skip = {"the", "a", "an", "at", "in", "of", "and", "or", "who", "what",
                "when", "gold", "silver", "bronze", "medal", "event", "olympics",
                "olympic", "summer", "winter", "held", "immediately", "before",
                "after", "previous", "next", sport.lower() if sport else ""}
        # Include Unicode (épée, etc.) and 2-char tokens (68, kg, 80) for fine discrimination
        q_words = [w for w in re.findall(r"[a-z0-9'À-ÿ]+", question.lower())
                   if w not in skip and len(w) >= 2]
        # Score: word-boundary match; penalise "+" prefix (e.g. "+80 kg" when asking "80 kg")
        def title_score(e):
            title_lower = e.get("title", "").lower()
            score = sum(1 for w in q_words
                        if re.search(rf'(?<!\+)\b{re.escape(w)}\b', title_lower))
            # Extra penalty: "+" events when question has no "+"
            if "+" not in question and "+" in title_lower:
                score -= 0.5
            return score
        best_match = max(anchor_candidates, key=title_score)
        if title_score(best_match) > 0:
            anchor = best_match

    # Follow edge
    anchor_vid = anchor["vid"]
    if direction == "PREV":
        target_vid = get_prev_edition(tg, anchor_vid)
    else:
        target_vid = get_next_edition(tg, anchor_vid)

    graph_path += f" → anchor={anchor_vid} → {direction}_EDITION → {target_vid}"

    if not target_vid:
        # Fall back: look for events of same sport/season 4 years earlier/later
        offset = -4 if direction == "PREV" else 4
        fb_candidates = filter_events(events, year=year + offset, season=season, sport=sport)
        if fb_candidates:
            target_vid = fb_candidates[0]["vid"]
            graph_path += f" (edge missing, fallback year {year+offset})"

    if not target_vid:
        return None, [anchor_vid], graph_path, [anchor_vid]

    doc_text = corpus.get_text(target_vid, max_chars=1500)
    gold = parse_medalist_name(doc_text, "gold")
    # Last-resort: pull the first capitalized full name from the document
    if not gold:
        gold = _extract_name_from_text(doc_text)
    graph_path += f" → gold={gold}"

    return gold, [anchor_vid, target_vid], graph_path, [target_vid]


def query_multi_hop(tg: TGClient, events: list[dict], corpus: Corpus,
                    question: str,
                    index: "EmbeddingIndex | None" = None) -> tuple[str, list[str], str]:
    """
    Multi-hop: find event by venue + date, return gold medalist.
    When multiple events match, use vector search to disambiguate.
    """
    venue_hint = extract_venue_hint(question)
    date_hint  = extract_date_hint(question)
    year       = extract_year(question)

    candidates = filter_events(events, year=year) if year else list(events)
    graph_path = f"Filter: year={year} → {len(candidates)} events"

    if venue_hint or date_hint:
        candidates = match_event_by_venue_date(candidates, venue_hint, date_hint)
        graph_path += f" → venue={venue_hint}, date={date_hint} → {len(candidates)} events"

    if not candidates:
        return None, [], graph_path

    # Disambiguate when multiple events match the same venue+date
    if len(candidates) > 1 and index is not None:
        hits = index.search(question, top_k=15)
        scores = {doc_id: score for doc_id, score in hits}
        # Candidates that appear in vector search results, ranked by score
        vec_matches = sorted(
            [c for c in candidates if c["vid"] in scores],
            key=lambda c: scores[c["vid"]],
            reverse=True,
        )
        if vec_matches:
            candidates = vec_matches
            graph_path += f" → vector disambiguated ({len(vec_matches)} candidates kept)"

    target = candidates[0]
    doc_text = corpus.get_text(target["vid"], max_chars=1500)
    gold = parse_medalist_name(doc_text, "gold")
    if not gold:
        gold = _extract_name_from_text(doc_text)
    graph_path += f" → event={target.get('title', target['vid'])[:60]} → gold={gold}"

    return gold, [target["vid"]], graph_path, [target["vid"]]


def _title_word_score(fragment: str, event: dict) -> int:
    """Score how many words from fragment appear as whole words in event title."""
    frag_norm = re.sub(r'[^\w]', ' ', fragment.lower())
    title_norm = re.sub(r'[^\w]', ' ', event.get("title", "").lower())
    words = [w for w in frag_norm.split() if len(w) > 1]
    return sum(1 for w in words if re.search(rf'\b{re.escape(w)}\b', title_norm))


def query_lookup(tg: TGClient, events: list[dict], corpus: Corpus,
                 question: str) -> tuple[str, list[str], str, list[str]]:
    """
    Lookup: find event by title fragment embedded in question, return nations/competitors.

    All lookup questions follow the pattern:
      'How many nations competed in [Full Event Title]?'
    """
    q_lower = question.lower()

    # Nations / competitors count: question embeds the full event title
    m = re.search(r'competed in (.+?)(?:\?|$)', question, re.IGNORECASE)
    if m:
        fragment = m.group(1).strip()
        # Score every event by title word overlap with the fragment
        best = max(events, key=lambda e: _title_word_score(fragment, e))
        best_score = _title_word_score(fragment, best)

        if best_score >= 3:  # require at least 3 words matched to avoid false positives
            if "nation" in q_lower or "countr" in q_lower:
                nations = best.get("nations")
                if nations is not None:
                    graph_path = (f"Title match ({best_score} words): "
                                  f"'{best['title']}' → nations={nations}")
                    return str(nations), [best["vid"]], graph_path, [best["vid"]]
                # Fallback: parse from infobox
                doc_text = corpus.get_text(best["vid"], max_chars=1500)
                nations = parse_infobox_field(doc_text, "nations")
                graph_path = (f"Title match: '{best['title']}' → "
                              f"nations={nations} (from infobox)")
                return nations, [best["vid"]], graph_path, [best["vid"]]
            elif "competitor" in q_lower or "participant" in q_lower:
                competitors = best.get("competitors")
                graph_path = f"Title match: '{best['title']}' → competitors={competitors}"
                return str(competitors) if competitors else None, [best["vid"]], graph_path, [best["vid"]]

    # Gold/silver/bronze medalist lookup
    year   = extract_year(question)
    season = extract_season(question)
    known_sports = {e.get("sport", "") for e in events if e.get("sport")}
    sport  = find_sport_match(question, known_sports)

    candidates = filter_events(events, year=year, season=season, sport=sport)
    graph_path = f"Lookup: year={year}, season={season}, sport={sport} → {len(candidates)} events"

    if not candidates:
        return None, [], graph_path, []

    skip = {"the", "a", "an", "at", "in", "of", "and", "or", "who", "what",
            "gold", "silver", "bronze", "medal", "event", "olympics", "olympic",
            "summer", "winter", "won", sport.lower() if sport else ""}
    q_words = [w for w in re.findall(r"[a-z0-9']+", question.lower())
               if w not in skip and len(w) > 2]
    def title_score(e):
        title_lower = e.get("title", "").lower()
        return sum(1 for w in q_words
                   if re.search(rf'\b{re.escape(w)}\b', title_lower))
    target = max(candidates, key=title_score)
    doc_text = corpus.get_text(target["vid"], max_chars=1500)

    if "gold" in q_lower or "winner" in q_lower or "won" in q_lower:
        answer = parse_medalist_name(doc_text, "gold")
    elif "silver" in q_lower:
        answer = parse_medalist_name(doc_text, "silver")
    elif "bronze" in q_lower:
        answer = parse_medalist_name(doc_text, "bronze")
    elif "venue" in q_lower or "held" in q_lower:
        answer = target.get("venue_name") or parse_infobox_field(doc_text, "venue")
    elif "competitor" in q_lower or "participant" in q_lower:
        answer = str(target.get("competitors") or "")
    elif "nation" in q_lower or "countr" in q_lower:
        answer = str(target.get("nations") or "")
    else:
        answer = parse_medalist_name(doc_text, "gold")

    graph_path += f" → answer={answer}"
    return answer, [target["vid"]], graph_path, [target["vid"]]


# ── Fallback: vector search + LLM ────────────────────────────

_GRAPHRAG_PROMPT = """\
Documents:
{context}

Question: {question}
{instruction}
Answer:"""


def _answer_instruction(question: str) -> str:
    """Build a type-specific LLM instruction so it doesn't hallucinate numbers for name questions."""
    q = question.lower()
    if any(w in q for w in ("who ", "winner", "won ", "gold medal", "silver medal", "bronze medal")):
        return "Give ONLY the person's full name. Do not give a number."
    if any(w in q for w in ("summarize", "summarise", "summary", "describe", "overview", "tell me about")):
        return "Write 2–3 sentences summarising the relevant information."
    if any(w in q for w in ("how many", "count", "number of")):
        return "Give ONLY the number."
    if any(w in q for w in ("which event", "what event")):
        return "Give ONLY the event name."
    return "Give the shortest possible answer — a name, number, or phrase."


def fallback_vector(index: EmbeddingIndex, corpus: Corpus,
                    question: str, top_k: int = 5) -> tuple[str, list[str], int]:
    """Vector search fallback when graph traversal finds nothing.

    Uses chat mode (llm_call) with a strict system prompt so qwen3:4b
    does not leak reasoning and respects the answer-type instruction.
    """
    hits = index.search(question, top_k=top_k)
    parts = []
    for i, (doc_id, _) in enumerate(hits, 1):
        title = corpus.get_title(doc_id)
        text  = corpus.get_text(doc_id, max_chars=800)
        parts.append(f"[Doc {i} | {title}]\n{text}")
    context = "\n\n---\n\n".join(parts)

    system = (
        "You are a precise Olympic data assistant. "
        "Answer using ONLY the provided documents. "
        "Never reveal your reasoning. Never output a number when a person's name is requested. "
        + _answer_instruction(question)
    )
    raw, tokens = llm_call(
        prompt=f"Documents:\n{context}\n\nQuestion: {question}",
        max_tokens=150,
        system=system,
    )
    answer = extract_final_answer(raw)
    return answer, [doc_id for doc_id, _ in hits], tokens


# ── Summary query ─────────────────────────────────────────────

def query_summary(tg: TGClient, events: list[dict], corpus: Corpus,
                  question: str, index: "EmbeddingIndex") -> tuple[str, list[str], str, list[str]]:
    """Summary: build a factual paragraph directly from structured data — no LLM needed.

    This avoids qwen3:4b's tendency to output reasoning preamble even when
    instructed not to. The structured data already contains all the facts.
    """
    year   = extract_year(question)
    season = extract_season(question)
    known_sports = {e.get("sport", "") for e in events if e.get("sport")}
    sport  = find_sport_match(question, known_sports)

    filtered = filter_events(events, year=year, season=season, sport=sport)
    graph_path = f"Summary: year={year}, season={season}, sport={sport} → {len(filtered)} events"

    if not filtered:
        return None, [], graph_path, []

    n = len(filtered)
    comp_vals = sorted([e["competitors"] for e in filtered if e.get("competitors")])
    nat_vals  = sorted([e["nations"]     for e in filtered if e.get("nations")])
    venues    = list(dict.fromkeys(
        e.get("venue_name", "") for e in filtered if e.get("venue_name")
    ))

    sport_label  = sport  or "sporting"
    season_label = f"{season} " if season else ""

    parts = [f"The {year} {season_label}Olympics featured {n} {sport_label} events."]

    if comp_vals:
        total = sum(comp_vals)
        parts.append(
            f"Events ranged from {comp_vals[0]} to {comp_vals[-1]} competitors "
            f"({total:,} total participations across all {sport_label} events)."
        )

    if nat_vals:
        parts.append(
            f"Between {nat_vals[0]} and {nat_vals[-1]} nations competed per event."
        )
    elif venues:
        venue_str = " and ".join(venues[:2]) if len(venues) > 1 else venues[0]
        parts.append(f"Events were held at {venue_str}.")

    if venues and nat_vals:
        venue_str = " and ".join(venues[:2]) if len(venues) > 1 else venues[0]
        parts.append(f"Most events took place at {venue_str}.")

    answer = " ".join(parts[:3])   # cap at 3 sentences
    all_ids = [e["vid"] for e in filtered]
    return answer, all_ids, graph_path, all_ids[:3]


# ── Main pipeline ─────────────────────────────────────────────

class GraphRAGPipeline:
    def __init__(self):
        self.corpus = Corpus()
        self.index  = EmbeddingIndex()
        self.tg     = None
        self.events = []
        self._tg_available = False
        try:
            self.tg = TGClient()
            print("Loading OlympicEvent vertices from TigerGraph ...", end=" ", flush=True)
            self.events = get_all_events(self.tg)
            print(f"{len(self.events)} events loaded.")
            self._tg_available = True
        except Exception as e:
            print(f"\n  WARNING: TigerGraph unavailable ({e})")
            print("  Running in vector-fallback-only mode.")
            self._tg_available = False

    def answer(self, question: str, qid: str = "", qtype: str = "",
               gold: list[str] = None) -> dict:
        t0 = time.time()
        graph_answer = None
        doc_ids      = []
        graph_path   = ""
        used_fallback = False

        answer_ids: list = []

        if self._tg_available and self.events:
            try:
                if qtype == "aggregation":
                    graph_answer, doc_ids, graph_path, answer_ids = query_aggregation(
                        self.tg, self.events, self.corpus, question)
                elif qtype == "superlative":
                    graph_answer, doc_ids, graph_path, answer_ids = query_superlative(
                        self.tg, self.events, self.corpus, question)
                elif qtype == "temporal":
                    graph_answer, doc_ids, graph_path, answer_ids = query_temporal(
                        self.tg, self.events, self.corpus, question)
                elif qtype == "multi_hop":
                    graph_answer, doc_ids, graph_path, answer_ids = query_multi_hop(
                        self.tg, self.events, self.corpus, question, self.index)
                elif qtype == "summary":
                    graph_answer, doc_ids, graph_path, answer_ids = query_summary(
                        self.tg, self.events, self.corpus, question, self.index)
                else:  # lookup
                    graph_answer, doc_ids, graph_path, answer_ids = query_lookup(
                        self.tg, self.events, self.corpus, question)
            except Exception as e:
                graph_path = f"Graph query error: {e}"
        else:
            graph_path = "TigerGraph unavailable, using vector fallback"

        # Fallback if graph returned nothing
        fb_tokens = 0
        if not graph_answer:
            fb_answer, fb_ids, fb_tokens = fallback_vector(self.index, self.corpus, question)
            graph_answer = fb_answer
            doc_ids.extend(fb_ids)
            answer_ids = fb_ids[:1]
            graph_path += " → [FALLBACK: vector search]"
            used_fallback = True

        # Post-validation: if question asks "who/winner/won" but answer starts with a digit → wrong
        if graph_answer:
            q_lower = question.lower()
            if any(w in q_lower for w in ("who ", "who's", "winner", "won the", "won gold", "won ")):
                if re.match(r'^\d', graph_answer.strip()):
                    graph_answer = "Information not found for this question"

        elapsed = time.time() - t0
        return make_result(
            pipeline     = "GraphRAG",
            qid          = qid,
            question     = question,
            qtype        = qtype,
            answer       = graph_answer or "Unknown",
            gold         = gold or [],
            retrieved_ids= doc_ids,
            tokens       = fb_tokens,
            elapsed      = elapsed,
            extra        = {
                "graph_path"     : graph_path,
                "used_fallback"  : used_fallback,
                "answer_node_ids": answer_ids,
                "query_trace"    : _build_query_trace(qtype, graph_path, question,
                                                      graph_answer or "Unknown", used_fallback),
            },
        )


def _build_query_trace(qtype: str, graph_path: str, question: str,
                       answer: str, used_fallback: bool) -> list[dict]:
    """Build a structured REST++-style execution trace from pipeline metadata."""
    G = "OlympicsGraph"
    trace = [{"op": "SOURCE", "detail": "TigerGraph Savanna · 2,187 OlympicEvent vertices", "endpoint": None}]

    raw = [s.strip() for s in re.split(r'\s*→\s*', graph_path) if s.strip()]

    op_for = {
        "aggregation": ["FILTER", "COUNT", "COUNT", "RETURN"],
        "superlative": ["FILTER", "SORT",  "INFOBOX"],
        "temporal":    ["FILTER", "EDGE",  "INFOBOX"],
        "lookup":      ["FILTER", "MATCH", "INFOBOX"],
        "multi_hop":   ["FILTER", "MATCH", "VECTOR", "INFOBOX"],
        "summary":     ["FILTER", "BUILD"],
    }
    ops = op_for.get(qtype, ["FILTER"] + ["STEP"] * 10)

    endpoint_for = {
        "FILTER":  f"GET /graph/{G}/vertices/OlympicEvent",
        "EDGE":    f"GET /graph/{G}/edges/OlympicEvent/{{vid}}/PREV_EDITION",
        "INFOBOX": "corpus.get_text(vid, max_chars=1500)",
        "VECTOR":  "ollama.embed(model='nomic-embed-text')",
        "FALLBACK":"ollama.embed(model='nomic-embed-text') · cosine similarity",
    }

    for i, step in enumerate(raw):
        op = ops[i] if i < len(ops) else "STEP"
        # auto-detect EDGE steps by content
        if "PREV_EDITION" in step or "NEXT_EDITION" in step:
            op = "EDGE"
            vid_m = re.search(r'anchor=([\w]+)', step)
            vid = vid_m.group(1) if vid_m else "{vid}"
            direction = "PREV_EDITION" if "PREV" in step else "NEXT_EDITION"
            endpoint_for["EDGE"] = f"GET /graph/{G}/edges/OlympicEvent/{vid}/{direction}"
        trace.append({"op": op, "detail": step, "endpoint": endpoint_for.get(op)})

    if used_fallback:
        trace.append({"op": "FALLBACK", "detail": "graph returned no result · vector similarity search",
                      "endpoint": endpoint_for["FALLBACK"]})

    if answer and answer not in ("Unknown", "Information not found for this question"):
        trace.append({"op": "RETURN", "detail": f"ANSWER: {answer}", "endpoint": None})

    return trace


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

    pipeline = GraphRAGPipeline()
    results  = []
    type_scores = {}

    print(f"\nRunning GraphRAG on {len(questions)} questions ...")
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

        em  = result["exact_match"]
        cg  = result["contains_gold"]
        fb  = "FB" if result["used_fallback"] else "G "
        print(f"[{i:3d}/{len(questions)}] {qid} ({qtype:<12}) "
              f"{fb} EM={em} CG={cg} t={result['elapsed_sec']:5.1f}s | "
              f"pred: {result['predicted_answer'][:40]!r}")

    print("\n" + "=" * 70)
    print("GRAPHRAG BENCHMARK SUMMARY")
    print("=" * 70)
    overall_em = sum(r["exact_match"]   for r in results) / len(results)
    overall_cg = sum(r["contains_gold"] for r in results) / len(results)
    print(f"Overall  exact_match={overall_em:.1%}  contains_gold={overall_cg:.1%}")
    print()
    print(f"{'Type':<15} {'N':>4}  {'ExactMatch':>10}  {'ContainsGold':>12}  {'FBRate':>7}")
    print("-" * 55)
    for qtype, s in sorted(type_scores.items()):
        n  = s["total"]
        em = s["exact_match"] / n
        cg = s["contains_gold"] / n
        fb = sum(1 for r in results if r["qtype"] == qtype and r["used_fallback"]) / n
        print(f"{qtype:<15} {n:>4}  {em:>10.1%}  {cg:>12.1%}  {fb:>7.1%}")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    payload = {
        "pipeline": "GraphRAG",
        "n_questions": len(results),
        "overall": {
            "exact_match"  : round(overall_em, 4),
            "contains_gold": round(overall_cg, 4),
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
