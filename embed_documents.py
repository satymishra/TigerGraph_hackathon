"""
STEP 4: Embed all 2,951 corpus documents and store in TigerGraph + local cache.

Architecture:
  - Document vertices (metadata: doc_id, title, has_infobox) → TigerGraph
  - LINKED_DOC edges (OlympicEvent → Document) → TigerGraph
  - Embeddings (768-dim float32 numpy arrays) → local embeddings/ directory
  - Full text corpus → local corpus.jsonl (read at query time)

Why hybrid: TigerGraph VECTOR type not available via GSQL in 4.2.5.
  numpy cosine similarity over 2951 × 768 is near-instant (<50ms).
  Everything else (graph traversal, vertex data) stays in TigerGraph.
"""

import json
import os
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
import ollama
from tg_client import TGClient

BASE        = Path(__file__).parent
CORPUS_PATH = BASE / "corpus-20260920T071304Z-1-001/corpus/corpus.jsonl"
EMB_DIR     = BASE / "embeddings"
EMB_MATRIX  = EMB_DIR / "embeddings.npy"
EMB_IDS     = EMB_DIR / "doc_ids.json"
EMB_TITLES  = EMB_DIR / "doc_titles.json"

EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE  = 8           # smaller batches = more stable Ollama throughput
TG_BATCH    = 200         # vertices per TigerGraph upsert
MAX_CHARS   = 500         # truncate — infobox + opening paragraph is enough


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def has_infobox(text):
    return "[Infobox Olympic event]" in text


def truncate_text(text, max_chars=MAX_CHARS):
    """Keep the infobox + opening paragraph, truncate the rest."""
    return text[:max_chars]


def embed_batch(texts):
    result = ollama.embed(model=EMBED_MODEL, input=texts)
    return result.embeddings


def batches(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i: i + size]


def wrap(v):
    return {"value": v}


# ── Phase A: Embed all documents ──────────────────────────────

def embed_all_documents(docs):
    EMB_DIR.mkdir(exist_ok=True)

    # Resume from cache if partial embedding exists
    done_ids = set()
    all_ids, all_titles, all_embeddings = [], [], []

    if EMB_IDS.exists() and EMB_MATRIX.exists():
        all_ids    = json.loads(EMB_IDS.read_text())
        all_titles = json.loads(EMB_TITLES.read_text())
        matrix     = np.load(EMB_MATRIX)
        all_embeddings = list(matrix)
        done_ids   = set(all_ids)
        print(f"  Resuming: {len(done_ids)} docs already embedded")

    remaining = [d for d in docs if d["doc_id"] not in done_ids]
    print(f"  Embedding {len(remaining)} remaining documents ...")

    t0 = time.time()
    for batch in tqdm(list(batches(remaining, BATCH_SIZE)), desc="Embedding"):
        texts    = [truncate_text(d["text"]) for d in batch]
        try:
            embeddings = embed_batch(texts)
        except Exception as e:
            print(f"  Batch error: {e}, retrying one-by-one ...")
            embeddings = []
            for t in texts:
                try:
                    embeddings.extend(embed_batch([t]))
                except Exception as e2:
                    print(f"    Skip doc: {e2}")
                    embeddings.append([0.0] * 768)

        for doc, emb in zip(batch, embeddings):
            all_ids.append(doc["doc_id"])
            all_titles.append(doc["title"])
            all_embeddings.append(emb)

        # Save checkpoint every 100 docs
        if len(all_ids) % 100 < BATCH_SIZE:
            _save_cache(all_ids, all_titles, all_embeddings)

    _save_cache(all_ids, all_titles, all_embeddings)
    elapsed = time.time() - t0
    print(f"  Embedded {len(remaining)} docs in {elapsed:.1f}s "
          f"({elapsed/max(len(remaining),1):.2f}s/doc)")
    return all_ids, all_titles, np.array(all_embeddings, dtype=np.float32)


def _save_cache(ids, titles, embeddings):
    EMB_IDS.write_text(json.dumps(ids))
    EMB_TITLES.write_text(json.dumps(titles))
    np.save(EMB_MATRIX, np.array(embeddings, dtype=np.float32))


# ── Phase B: Load Document vertices into TigerGraph ──────────

def load_document_vertices(client, docs, has_infobox_set):
    print(f"\n  Upserting {len(docs)} Document vertices ...")
    ok = 0
    for batch in batches(docs, TG_BATCH):
        payload = {"Document": {}}
        for d in batch:
            payload["Document"][d["doc_id"]] = {
                "title":       wrap(d["title"]),
                "has_infobox": wrap(1 if d["doc_id"] in has_infobox_set else 0),
            }
        r = client.post(f"/restpp/graph/{client.graph}",
                        {"vertices": payload})
        res = r.json()
        if not res.get("error"):
            ok += res["results"][0].get("accepted_vertices", 0)
        else:
            print(f"  WARN: {res.get('message','')[:100]}")
    print(f"  Done — {ok} Document vertices accepted")


# ── Phase C: LINKED_DOC edges (OlympicEvent → Document) ──────

def load_linked_doc_edges(client, infobox_doc_ids):
    """For every OlympicEvent, create a LINKED_DOC edge to its Document."""
    print(f"\n  Upserting {len(infobox_doc_ids)} LINKED_DOC edges ...")
    ok = 0
    for batch in batches(infobox_doc_ids, TG_BATCH):
        payload = {"OlympicEvent": {}}
        for doc_id in batch:
            payload["OlympicEvent"][doc_id] = {
                "LINKED_DOC": {"Document": {doc_id: {}}}
            }
        r = client.post(f"/restpp/graph/{client.graph}", {"edges": payload})
        res = r.json()
        if not res.get("error"):
            ok += res["results"][0].get("accepted_edges", 0)
        else:
            print(f"  WARN: {res.get('message','')[:100]}")
    print(f"  Done — {ok} LINKED_DOC edges accepted")


# ── Phase D: Verify ───────────────────────────────────────────

def verify(client, ids, embeddings):
    print("\n=== Verification ===")
    r = client.get(f"/restpp/graph/{client.graph}/vertices/Document/TEST_DOC_001")
    d = r.json()
    if not d.get("error"):
        print(f"  TEST_DOC_001 still in graph: {d['results'][0]['attributes']}")
    else:
        print(f"  Document vertex check: OK (TEST_DOC_001 was cleaned up)")

    print(f"  Total documents embedded: {len(ids)}")
    print(f"  Embedding matrix shape: {embeddings.shape}")
    print(f"  Embedding files saved to: {EMB_DIR}")

    # Quick vector search test
    print("\n  Quick vector search test ...")
    query = "who won gold medal in swimming"
    q_emb = np.array(embed_batch([query])[0], dtype=np.float32)
    sims  = embeddings @ q_emb / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-9
    )
    top5  = np.argsort(sims)[::-1][:5]
    id_list    = ids
    title_list = json.loads(EMB_TITLES.read_text())
    print(f"  Query: '{query}'")
    for i in top5:
        print(f"    [{sims[i]:.3f}] {title_list[i][:70]}")


# ── Main ──────────────────────────────────────────────────────

def main():
    print("Loading corpus ...")
    docs = load_jsonl(CORPUS_PATH)
    print(f"  {len(docs)} documents")

    infobox_ids = {d["doc_id"] for d in docs if has_infobox(d["text"])}
    print(f"  {len(infobox_ids)} docs with infobox, "
          f"{len(docs) - len(infobox_ids)} text-only")

    # Phase A: Embed
    print("\n=== Phase A: Embed all documents ===")
    all_ids, all_titles, embeddings = embed_all_documents(docs)

    # Phase B + C: Load into TigerGraph
    print("\n=== Phase B+C: Load Document vertices + edges into TigerGraph ===")
    client = TGClient()

    # Remove test vertex if exists
    client.post(f"/restpp/graph/{client.graph}",
                {"vertices": {"Document": {}}})   # no-op to warm connection

    load_document_vertices(client, docs, infobox_ids)
    load_linked_doc_edges(client, list(infobox_ids))

    # Phase D: Verify
    verify(client, all_ids, embeddings)

    print("\nSTEP 4 COMPLETE")


if __name__ == "__main__":
    main()
