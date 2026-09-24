"""
Step 8: Benchmark comparison dashboard.

Reads results from:
  results/rag_results.json
  results/graphrag_results.json
  results/agentic_results.json

Produces:
  results/comparison_summary.txt  — plain text table
  results/comparison.html         — HTML report with charts

Usage:
  python compare_benchmarks.py
"""

import json
from pathlib import Path

BASE         = Path(__file__).parent
RESULTS_DIR  = BASE / "results"
RAG_FILE     = RESULTS_DIR / "rag_results.json"
GR_FILE      = RESULTS_DIR / "graphrag_results.json"
AG_FILE      = RESULTS_DIR / "agentic_results.json"


def load_results(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def per_question_type_table(results_list: list[tuple[str, dict]]) -> str:
    """Build a comparison table per question type."""
    # Collect all types
    all_types = set()
    for name, data in results_list:
        if data:
            all_types.update(data.get("by_type", {}).keys())
    all_types = sorted(all_types)

    lines = []
    # Header
    lines.append(f"{'Type':<14}" + "".join(f"  {name[:10]:>12}  {'CG':>6}" for name, _ in results_list))
    lines.append("-" * (14 + len(results_list) * 22))

    for qtype in all_types:
        row = f"{qtype:<14}"
        for name, data in results_list:
            if data and qtype in data.get("by_type", {}):
                t = data["by_type"][qtype]
                em = t.get("exact_match", 0)
                cg = t.get("contains_gold", 0)
                n  = t.get("n", 0)
                row += f"  {em:>10.1%}  {cg:>6.1%}"
            else:
                row += "  " + " " * 10 + "  " + " " * 6
        lines.append(row)

    return "\n".join(lines)


def print_comparison(results_list: list[tuple[str, dict]]) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("BENCHMARK COMPARISON: RAG vs GraphRAG vs Agentic GraphRAG")
    lines.append("=" * 70)
    lines.append("")

    # Overall scores
    lines.append(f"{'Pipeline':<22}  {'ExactMatch':>10}  {'ContainsGold':>12}  {'Avg Tokens':>10}")
    lines.append("-" * 60)
    for name, data in results_list:
        if data:
            em  = data.get("overall", {}).get("exact_match", 0)
            cg  = data.get("overall", {}).get("contains_gold", 0)
            tok = data.get("overall", {}).get("avg_tokens") or (
                  data.get("overall", {}).get("total_tokens", 0) /
                  max(data.get("n_questions", 1), 1))
            lines.append(f"{name:<22}  {em:>10.1%}  {cg:>12.1%}  {tok:>10.1f}")
        else:
            lines.append(f"{name:<22}  {'(not run)':>10}")
    lines.append("")

    # Per-type
    lines.append("Per-question-type breakdown (ExactMatch  ContainsGold):")
    lines.append("")
    header = f"{'Type':<14}"
    for name, _ in results_list:
        short = name[:10]
        header += f"  {short:>10}  {'CG':>6}"
    lines.append(header)
    lines.append("-" * (14 + len(results_list) * 20))

    all_types = sorted(set(
        qtype
        for _, data in results_list
        if data
        for qtype in data.get("by_type", {})
    ))
    for qtype in all_types:
        row = f"{qtype:<14}"
        for name, data in results_list:
            if data and qtype in data.get("by_type", {}):
                t = data["by_type"][qtype]
                em = t.get("exact_match", 0)
                cg = t.get("contains_gold", 0)
                row += f"  {em:>10.1%}  {cg:>6.1%}"
            else:
                row += "  " + " " * 10 + "  " + " " * 6
        lines.append(row)

    return "\n".join(lines)


def build_html(results_list: list[tuple[str, dict]], text_summary: str) -> str:
    """Build a simple HTML report."""
    pipeline_colors = {
        "RAG"             : "#4285f4",
        "GraphRAG"        : "#34a853",
        "AgenticGraphRAG" : "#ea4335",
    }

    def bar(value, color, width=300):
        px = int(value * width)
        return (f'<div style="display:inline-block;background:{color};'
                f'width:{px}px;height:16px;vertical-align:middle;"></div>'
                f'<span style="margin-left:4px">{value:.1%}</span>')

    rows = []
    available = [(n, d) for n, d in results_list if d]

    # Overall comparison table
    rows.append("<h2>Overall Accuracy</h2>")
    rows.append("<table border=1 cellpadding=6 cellspacing=0>")
    rows.append("<tr><th>Pipeline</th><th>ExactMatch</th><th>ContainsGold</th><th>N questions</th></tr>")
    for name, data in available:
        em  = data["overall"].get("exact_match", 0)
        cg  = data["overall"].get("contains_gold", 0)
        n   = data.get("n_questions", 0)
        col = pipeline_colors.get(name, "#999")
        rows.append(f"<tr><td><b>{name}</b></td>"
                    f"<td>{bar(em, col)}</td>"
                    f"<td>{bar(cg, '#aaa')}</td>"
                    f"<td>{n}</td></tr>")
    rows.append("</table>")

    # Per-type
    rows.append("<h2>Per Question Type (ExactMatch)</h2>")
    rows.append("<table border=1 cellpadding=6 cellspacing=0>")
    header = "<tr><th>Type</th>" + "".join(f"<th>{n}</th>" for n, _ in available) + "</tr>"
    rows.append(header)

    all_types = sorted(set(
        qt for _, d in available for qt in d.get("by_type", {})))
    for qt in all_types:
        row = f"<tr><td>{qt}</td>"
        for name, data in available:
            t = data.get("by_type", {}).get(qt, {})
            em  = t.get("exact_match", 0)
            col = pipeline_colors.get(name, "#999")
            row += f"<td>{bar(em, col, 200)}</td>"
        row += "</tr>"
        rows.append(row)
    rows.append("</table>")

    # Per-question detail (show all 3 pipelines side by side)
    rows.append("<h2>Per-Question Detail</h2>")
    rows.append("<table border=1 cellpadding=4 cellspacing=0>")
    header = ("<tr><th>QID</th><th>Type</th><th>Question</th>"
              + "".join(f"<th>{n} pred</th><th>EM</th>" for n, _ in available)
              + "<th>Gold</th></tr>")
    rows.append(header)

    # Build index by qid for each pipeline
    by_pipeline = {}
    for name, data in available:
        by_pipeline[name] = {r["qid"]: r for r in data.get("results", [])}

    all_qids = []
    for _, data in available:
        for r in data.get("results", []):
            if r["qid"] not in all_qids:
                all_qids.append(r["qid"])

    for qid in all_qids:
        first = next((by_pipeline[n].get(qid) for n, _ in available
                      if by_pipeline.get(n, {}).get(qid)), None)
        if not first:
            continue
        row = (f"<tr><td>{qid}</td><td>{first['qtype']}</td>"
               f"<td>{first['question'][:60]}...</td>")
        for name, _ in available:
            r = by_pipeline.get(name, {}).get(qid)
            if r:
                em_col = "#c8e6c9" if r["exact_match"] else "#ffcdd2"
                row += (f"<td style='background:{em_col}'>{r['predicted_answer'][:40]}</td>"
                        f"<td style='background:{em_col}'>{r['exact_match']}</td>")
            else:
                row += "<td>-</td><td>-</td>"
        gold = first.get("gold_answer", [])
        row += f"<td>{', '.join(gold[:2])}</td></tr>"
        rows.append(row)
    rows.append("</table>")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>GraphRAG Hackathon — Benchmark Comparison</title>
<style>
  body {{ font-family: sans-serif; margin: 20px; }}
  h1   {{ color: #333; }}
  h2   {{ color: #555; margin-top: 30px; }}
  table {{ border-collapse: collapse; margin-bottom: 20px; }}
  th   {{ background: #f0f0f0; }}
  pre  {{ background: #f8f8f8; padding: 12px; font-size: 12px; overflow-x: auto; }}
</style>
</head>
<body>
<h1>TigerGraph Agentic GraphRAG — Benchmark Comparison</h1>
<p>Dataset: Olympic Events (100 questions)</p>
{''.join(rows)}
<h2>Text Summary</h2>
<pre>{text_summary}</pre>
</body>
</html>"""
    return html


def main():
    rag = load_results(RAG_FILE)
    gr  = load_results(GR_FILE)
    ag  = load_results(AG_FILE)

    results_list = [
        ("RAG",             rag),
        ("GraphRAG",        gr),
        ("AgenticGraphRAG", ag),
    ]

    available = [(n, d) for n, d in results_list if d]
    if not available:
        print("No results files found. Run the benchmarks first.")
        return

    summary = print_comparison(results_list)
    print(summary)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "comparison_summary.txt").write_text(summary, encoding="utf-8")
    print(f"\nSummary saved to {RESULTS_DIR / 'comparison_summary.txt'}")

    html = build_html(results_list, summary)
    html_path = RESULTS_DIR / "comparison.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"HTML report saved to {html_path}")
    print(f"\nOpen: {html_path}")


if __name__ == "__main__":
    main()
