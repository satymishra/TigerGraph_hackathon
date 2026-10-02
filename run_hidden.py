"""
Run all three pipelines on eval_hidden.jsonl and save results.
Usage: python run_hidden.py [graphrag|agentic|rag]
       python run_hidden.py          # runs all three
"""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent
HIDDEN_FILE = ROOT / "questions-20260920T071303Z-1-001/questions/eval_hidden.jsonl"
OUT_DIR     = ROOT / "results"

def load_questions():
    with open(HIDDEN_FILE, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def run_pipeline(name: str, questions: list):
    print(f"\n{'='*60}")
    print(f"Running {name.upper()} on {len(questions)} hidden questions")
    print(f"{'='*60}")

    if name == "graphrag":
        from pipelines.graphrag_pipeline import GraphRAGPipeline
        pipeline = GraphRAGPipeline()
    elif name == "agentic":
        from pipelines.agentic_pipeline import AgenticGraphRAGPipeline
        pipeline = AgenticGraphRAGPipeline()
    elif name == "rag":
        from pipelines.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline()
    else:
        raise ValueError(f"Unknown pipeline: {name}")

    results = []
    for i, q in enumerate(questions, 1):
        qid      = q["qid"]
        question = q["question"]
        qtype    = q.get("qtype", "")

        print(f"[{i:02d}/50] {qid} ({qtype}) — {question[:70]}...")
        t0 = time.time()
        try:
            res = pipeline.answer(question, qid=qid, qtype=qtype, gold=[])
        except Exception as e:
            print(f"  ERROR: {e}")
            res = {"predicted_answer": "Error", "tokens_used": 0, "graph_path": "", "evidence_chain": []}

        elapsed = round(time.time() - t0, 2)

        record = {
            "pipeline":          name,
            "qid":               qid,
            "question":          question,
            "qtype":             qtype,
            "predicted_answer":  res.get("predicted_answer", ""),
            "tokens_used":       res.get("tokens_used", 0),
            "elapsed_sec":       elapsed,
            "graph_path":        res.get("graph_path") or res.get("evidence_chain") or "",
            "evidence_chain":    res.get("evidence_chain") or [],
            "used_fallback":     res.get("used_fallback", False),
            "retrieved_doc_ids": res.get("retrieved_doc_ids") or [],
        }
        results.append(record)
        print(f"  Answer: {record['predicted_answer']}  ({elapsed}s, {record['tokens_used']} tokens)")

    # Save
    out_path = OUT_DIR / f"{name}_hidden_results.json"
    avg_tokens = sum(r["tokens_used"] for r in results) / len(results)
    avg_time   = sum(r["elapsed_sec"]  for r in results) / len(results)

    output = {
        "pipeline":    name,
        "dataset":     "eval_hidden.jsonl",
        "total":       len(results),
        "avg_tokens":  round(avg_tokens, 1),
        "avg_elapsed": round(avg_time, 2),
        "results":     results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {out_path}")
    print(f"Avg tokens: {avg_tokens:.1f}  |  Avg time: {avg_time:.2f}s")
    return results


if __name__ == "__main__":
    questions = load_questions()
    targets   = sys.argv[1:] or ["graphrag", "agentic", "rag"]

    for name in targets:
        run_pipeline(name, questions)

    print("\nDone. Results saved to results/")
