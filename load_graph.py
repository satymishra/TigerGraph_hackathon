"""
STEP 3: Parse corpus and load knowledge graph into TigerGraph.
Uses TigerGraph 4.x REST++ format: attribute values wrapped in {"value": ...}

Loads:
  Vertices:  OlympicEvent, Medalist, Country, Venue
  Edges:     GOLD_IN, SILVER_IN, BRONZE_IN, HELD_AT,
             GOLD_COUNTRY, PREV_EDITION, NEXT_EDITION
"""

import json
import re
import time
from collections import defaultdict
from pathlib import Path

from tg_client import TGClient

BASE        = Path(__file__).parent
CORPUS_PATH = BASE / "corpus-20260920T071304Z-1-001/corpus/corpus.jsonl"
BATCH_SIZE  = 150


# ── Helpers ───────────────────────────────────────────────────

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_infobox(text):
    fields = {}
    in_box = False
    for raw in text.split("\n"):
        s = raw.strip()
        if s == "[Infobox Olympic event]":
            in_box = True
            continue
        if in_box:
            if not s:
                break
            if ":" in s:
                k, _, v = s.partition(":")
                fields[k.strip()] = v.strip()
    return fields


def parse_games(games_str):
    m = re.match(r"(\d{4})\s+(Summer|Winter)", games_str.strip())
    return (int(m.group(1)), m.group(2)) if m else (None, None)


def safe_int(s, cap=10000):
    try:
        v = int(str(s).strip().replace(",", ""))
        return v if 0 <= v <= cap else -1
    except (ValueError, TypeError):
        return -1


def extract_sport_event(title):
    m = re.search(r" at the \d{4} (?:Summer|Winter) Olympics", title)
    if not m:
        return None, None
    sport = title[: m.start()].strip()
    rest  = title[m.end():]
    dash  = re.search(r" [–\-] (.+)$", rest)
    event_suffix = dash.group(1).strip() if dash else rest.strip()
    return sport, event_suffix


def make_medalist_id(name, noc):
    return f"{name}|{noc}"


def batches(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i: i + size]


# ── TG 4.x format helpers ─────────────────────────────────────

def wrap_attrs(d):
    """Wrap all values in {'value': v} for TG 4.x REST++ upsert."""
    return {k: {"value": v} for k, v in d.items()}


def upsert(client, payload, label=""):
    r = client.post(f"/restpp/graph/{client.graph}", payload, timeout=60)
    if r.status_code != 200 or r.json().get("error"):
        msg = r.json().get("message", r.text[:200])
        print(f"  WARN [{label}]: {msg}")
        return 0
    res = r.json().get("results", [{}])[0]
    return res.get("accepted_vertices", 0) + res.get("accepted_edges", 0)


# ── Parse all documents ───────────────────────────────────────

def parse_documents(docs):
    events = []
    skipped = 0
    for doc in docs:
        fields = parse_infobox(doc["text"])
        if not fields:
            skipped += 1
            continue
        year, season = parse_games(fields.get("games", ""))
        if year is None:
            skipped += 1
            continue
        sport, event_suffix = extract_sport_event(doc["title"])
        events.append({
            "doc_id"      : doc["doc_id"],
            "title"       : doc["title"],
            "event_name"  : fields.get("event", ""),
            "games"       : fields.get("games", ""),
            "sport"       : sport or "",
            "season"      : season,
            "year"        : year,
            "venue_name"  : fields.get("venue", ""),
            "date_str"    : fields.get("date", fields.get("dates", "")),
            "competitors" : safe_int(fields.get("competitors", -1)),
            "nations"     : safe_int(fields.get("nations", -1)),
            "win_value"   : fields.get("win_value", ""),
            "url"         : doc.get("url", ""),
            "wikidata_qid": doc.get("wikidata_qid", ""),
            "_fields"     : fields,
            "_sport"      : sport,
            "_event_suffix": event_suffix,
        })
    print(f"  Parsed {len(events)} events, {skipped} docs skipped (no infobox)")
    return events


# ── Phase 1: Vertices ─────────────────────────────────────────

def load_event_vertices(client, events):
    skip = {"_fields", "_sport", "_event_suffix"}
    total = 0
    for batch in batches(events, BATCH_SIZE):
        payload = {"OlympicEvent": {}}
        for e in batch:
            vid   = e["doc_id"]
            attrs = {k: v for k, v in e.items() if k not in skip and k != "doc_id"}
            payload["OlympicEvent"][vid] = wrap_attrs(attrs)
        total += upsert(client, {"vertices": payload}, "OlympicEvent")
    print(f"  OlympicEvent: {total} accepted")


def load_medalists(client, events):
    seen = {}
    for e in events:
        f = e["_fields"]
        for nk, noc_k in [("gold","goldNOC"),("silver","silverNOC"),
                           ("bronze","bronzeNOC"),("bronze2","bronzeNOC2")]:
            name = f.get(nk, "").strip()
            noc  = f.get(noc_k, "").strip()
            if name and noc:
                mid = make_medalist_id(name, noc)
                seen[mid] = {"name": name, "noc_code": noc}

    total = 0
    for batch in batches(list(seen.items()), BATCH_SIZE):
        payload = {"Medalist": {}}
        for mid, attrs in batch:
            payload["Medalist"][mid] = wrap_attrs(attrs)
        total += upsert(client, {"vertices": payload}, "Medalist")
    print(f"  Medalist: {total} accepted ({len(seen)} unique)")


def load_countries(client, events):
    nocs = {e["_fields"].get(k, "").strip()
            for e in events
            for k in ("goldNOC","silverNOC","bronzeNOC","bronzeNOC2")} - {""}
    total = 0
    for batch in batches(list(nocs), BATCH_SIZE):
        payload = {"Country": {noc: {} for noc in batch}}
        total += upsert(client, {"vertices": payload}, "Country")
    print(f"  Country: {total} accepted ({len(nocs)} unique)")


def load_venues(client, events):
    venues = {e["venue_name"].strip() for e in events if e["venue_name"].strip()}
    total = 0
    for batch in batches(list(venues), BATCH_SIZE):
        payload = {"Venue": {v: {} for v in batch}}
        total += upsert(client, {"vertices": payload}, "Venue")
    print(f"  Venue: {total} accepted ({len(venues)} unique)")


# ── Phase 2: Edges ────────────────────────────────────────────

def load_medal_edges(client, events):
    configs = [
        ("gold",   "goldNOC",   "GOLD_IN"),
        ("silver", "silverNOC", "SILVER_IN"),
        ("bronze", "bronzeNOC", "BRONZE_IN"),
        ("bronze2","bronzeNOC2","BRONZE_IN"),
    ]
    for name_k, noc_k, etype in configs:
        edges = []
        for e in events:
            f    = e["_fields"]
            name = f.get(name_k, "").strip()
            noc  = f.get(noc_k, "").strip()
            if name and noc:
                edges.append((make_medalist_id(name, noc), e["doc_id"]))
        total = 0
        for batch in batches(edges, BATCH_SIZE):
            payload = {"Medalist": {}}
            for fid, tid in batch:
                if fid not in payload["Medalist"]:
                    payload["Medalist"][fid] = {etype: {"OlympicEvent": {}}}
                payload["Medalist"][fid][etype]["OlympicEvent"][tid] = {}
            total += upsert(client, {"edges": payload}, etype)
        print(f"  {etype} ({name_k}): {total} accepted")


def load_venue_edges(client, events):
    edges = [(e["doc_id"], e["venue_name"].strip())
             for e in events if e["venue_name"].strip()]
    total = 0
    for batch in batches(edges, BATCH_SIZE):
        payload = {"OlympicEvent": {}}
        for doc_id, venue in batch:
            payload["OlympicEvent"][doc_id] = {
                "HELD_AT": {"Venue": {venue: {}}}
            }
        total += upsert(client, {"edges": payload}, "HELD_AT")
    print(f"  HELD_AT: {total} accepted")


def load_gold_country_edges(client, events):
    edges = [(e["doc_id"], e["_fields"].get("goldNOC", "").strip())
             for e in events if e["_fields"].get("goldNOC", "").strip()]
    total = 0
    for batch in batches(edges, BATCH_SIZE):
        payload = {"OlympicEvent": {}}
        for doc_id, noc in batch:
            payload["OlympicEvent"][doc_id] = {
                "GOLD_COUNTRY": {"Country": {noc: {}}}
            }
        total += upsert(client, {"edges": payload}, "GOLD_COUNTRY")
    print(f"  GOLD_COUNTRY: {total} accepted")


# ── Phase 3: Temporal edges ───────────────────────────────────

def load_temporal_edges(client, events):
    lookup = defaultdict(dict)
    for e in events:
        key = (e["_sport"], e["_event_suffix"], e["season"])
        lookup[key][e["year"]] = e["doc_id"]

    prev_edges, next_edges = [], []
    for e in events:
        f   = e["_fields"]
        key = (e["_sport"], e["_event_suffix"], e["season"])
        for field, bucket in [("prev", prev_edges), ("next", next_edges)]:
            yr_str = f.get(field, "").strip()
            if re.fullmatch(r"\d{4}", yr_str):
                yr = int(yr_str)
                if yr in lookup[key]:
                    bucket.append((e["doc_id"], lookup[key][yr], yr))

    for etype, edge_list in [("PREV_EDITION", prev_edges),
                              ("NEXT_EDITION", next_edges)]:
        total = 0
        for batch in batches(edge_list, BATCH_SIZE):
            payload = {"OlympicEvent": {}}
            for fid, tid, yr in batch:
                attr_key = "prev_year" if etype == "PREV_EDITION" else "next_year"
                payload["OlympicEvent"][fid] = {
                    etype: {"OlympicEvent": {
                        tid: {attr_key: {"value": yr}}
                    }}
                }
            total += upsert(client, {"edges": payload}, etype)
        print(f"  {etype}: {total} accepted ({len(edge_list)} total)")


# ── Verify ────────────────────────────────────────────────────

def verify(client):
    print("\n=== Verification ===")
    # Look up a few known vertices
    for vtype, vid in [
        ("OlympicEvent", "Q303623"),
        ("OlympicEvent", "Q607635"),
    ]:
        r = client.get(f"/restpp/graph/{client.graph}/vertices/{vtype}/{vid}")
        d = r.json()
        if not d.get("error") and d.get("results"):
            a = d["results"][0]["attributes"]
            print(f"  {vtype} {vid}: title='{a.get('title','?')[:50]}' "
                  f"year={a.get('year')} competitors={a.get('competitors')}")
        else:
            print(f"  {vtype} {vid}: NOT FOUND — {d.get('message','')}")

    # Count via GSQL
    for vtype in ["OlympicEvent", "Medalist", "Country", "Venue"]:
        result = client.gsql(
            f"USE GRAPH {client.graph}\n"
            f"SELECT COUNT(*) FROM {vtype}"
        )
        # Extract number from result
        m = re.search(r'"count"\s*:\s*(\d+)', result)
        count = m.group(1) if m else "?"
        print(f"  {vtype} count: {count}")


# ── Main ──────────────────────────────────────────────────────

def main():
    print("Loading corpus ...")
    docs = load_jsonl(CORPUS_PATH)
    print(f"  {len(docs)} documents")

    print("\nParsing infoboxes ...")
    events = parse_documents(docs)

    client = TGClient()
    print(f"Connected to {client.graph}")

    t0 = time.time()

    print("\n=== Phase 1: Vertices ===")
    load_event_vertices(client, events)
    load_medalists(client, events)
    load_countries(client, events)
    load_venues(client, events)

    print("\n=== Phase 2: Edges ===")
    load_medal_edges(client, events)
    load_venue_edges(client, events)
    load_gold_country_edges(client, events)

    print("\n=== Phase 3: Temporal Edges ===")
    load_temporal_edges(client, events)

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    verify(client)
    print("\nSTEP 3 COMPLETE")


if __name__ == "__main__":
    main()
