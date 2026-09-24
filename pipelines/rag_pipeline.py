"""
Pipeline 1: RAG (Retrieval-Augmented Generation) — baseline.

Strategy:
  1. Embed the question with nomic-embed-text
  2. Cosine similarity search over all 2,951 document embeddings
  3. Take top-K docs, truncate to fit LLM context
  4. Ask qwen3:4b to answer from those docs only

Expected behaviour:
  - Good  on: lookup, simple multi_hop (1-2 docs needed)
  - Poor  on: aggregation, superlative (need 10-43 docs, RAG misses most)
  - Moderate on: temporal (may retrieve wrong year's docs)
"""

import json
import time
from pathlib import Path
from .shared import (
    Corpus, EmbeddingIndex, llm_complete, extract_final_answer, make_result, normalise
)

TOP_K        = 5    # docs retrieved per question
MAX_DOC_CHARS = 800  # chars per doc sent to LLM (infobox + opening, enough for facts)
RESULTS_PATH = Path(__file__).parent.parent / "results" / "rag_results.json"


RAG_PROMPT = """\
Documents:
{context}

Using ONLY the documents above, answer this question with the shortest possible answer (a name, number, or phrase — no explanation).

Question: {question}
Answer:"""


class RAGPipeline:
    def __init__(self, top_k: int = TOP_K):
        self.top_k  = top_k
        self.corpus = Corpus()
        self.index  = EmbeddingIndex()

    def retrieve(self, question: str) -> list[tuple[str, float]]:
        return self.index.search(question, top_k=self.top_k)

    def build_context(self, hits: list[tuple[str, float]]) -> str:
        parts = []
        for i, (doc_id, score) in enumerate(hits, 1):
            title = self.corpus.get_title(doc_id)
            text  = self.corpus.get_text(doc_id, max_chars=MAX_DOC_CHARS)
            parts.append(f"[Doc {i} | {title}]\n{text}")
        return "\n\n---\n\n".join(parts)

    def answer(self, question: str, qid: str = "", qtype: str = "",
               gold: list[str] = None) -> dict:
        t0 = time.time()

        hits    = self.retrieve(question)
        context = self.build_context(hits)
        prompt  = RAG_PROMPT.format(context=context, question=question)

        raw_text, tokens = llm_complete(prompt, max_tokens=350)
        answer_text = extract_final_answer(raw_text)
        elapsed = time.time() - t0

        return make_result(
            pipeline     = "RAG",
            qid          = qid,
            question     = question,
            qtype        = qtype,
            answer       = answer_text,
            gold         = gold or [],
            retrieved_ids= [doc_id for doc_id, _ in hits],
            tokens       = tokens,
            elapsed      = elapsed,
            extra        = {
                "top_k"            : self.top_k,
                "retrieval_scores" : [round(s, 4) for _, s in hits],
                "raw_llm_output"   : raw_text,
            },
        )


# ── Benchmark runner ──────────────────────────────────────────

def run_benchmark(questions_path: str = None, limit: int = None,
                  top_k: int = TOP_K) -> list[dict]:
    from pathlib import Path
    import json

    if questions_path is None:
        questions_path = (
            Path(__file__).parent.parent
            / "questions-20260920T071303Z-1-001/questions/eval_public.jsonl"
        )

    with open(questions_path, encoding="utf-8") as f:
        questions = [json.loads(l) for l in f if l.strip()]

    if limit:
        questions = questions[:limit]

    pipeline = RAGPipeline(top_k=top_k)
    results  = []

    # Per-type tracking
    type_scores = {}

    print(f"\nRunning RAG on {len(questions)} questions (top_k={top_k}) ...")
    print("-" * 70)

    for i, q in enumerate(questions, 1):
        qid   = q["qid"]
        qtype = q["qtype"]
        gold  = q.get("answer", [])

        result = pipeline.answer(
            question=q["question"],
            qid=qid,
            qtype=qtype,
            gold=gold,
        )
        results.append(result)

        # Track accuracy per type
        if qtype not in type_scores:
            type_scores[qtype] = {"total": 0, "contains_gold": 0, "exact_match": 0}
        type_scores[qtype]["total"]        += 1
        type_scores[qtype]["contains_gold"] += result["contains_gold"]
        type_scores[qtype]["exact_match"]   += result["exact_match"]

        # Progress line
        em  = result["exact_match"]
        cg  = result["contains_gold"]
        tok = result["tokens_used"]
        print(f"[{i:3d}/{len(questions)}] {qid} ({qtype:<12}) "
              f"EM={em} CG={cg} tok={tok:4d} | "
              f"pred: {result['predicted_answer'][:50]!r}")

    # Summary
    print("\n" + "=" * 70)
    print("RAG BENCHMARK SUMMARY")
    print("=" * 70)
    overall_em = sum(r["exact_match"]   for r in results) / len(results)
    overall_cg = sum(r["contains_gold"] for r in results) / len(results)
    overall_tok = sum(r["tokens_used"]  for r in results)
    avg_tok     = overall_tok / len(results)
    print(f"Overall  exact_match={overall_em:.1%}  contains_gold={overall_cg:.1%}  "
          f"avg_tokens={avg_tok:.0f}")
    print()
    print(f"{'Type':<15} {'N':>4}  {'ExactMatch':>10}  {'ContainsGold':>12}")
    print("-" * 45)
    for qtype, s in sorted(type_scores.items()):
        n  = s["total"]
        em = s["exact_match"] / n
        cg = s["contains_gold"] / n
        print(f"{qtype:<15} {n:>4}  {em:>10.1%}  {cg:>12.1%}")

    # Save results
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    payload = {
        "pipeline"    : "RAG",
        "top_k"       : top_k,
        "n_questions" : len(results),
        "overall"     : {
            "exact_match"  : round(overall_em, 4),
            "contains_gold": round(overall_cg, 4),
            "avg_tokens"   : round(avg_tok, 1),
            "total_tokens" : overall_tok,
        },
        "by_type"     : {
            t: {
                "exact_match"  : round(s["exact_match"] / s["total"], 4),
                "contains_gold": round(s["contains_gold"] / s["total"], 4),
                "n"            : s["total"],
            }
            for t, s in type_scores.items()
        },
        "results"     : results,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_benchmark(limit=limit)
