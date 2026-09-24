"""
Shared utilities across RAG / GraphRAG / Agentic GraphRAG pipelines.
  - Corpus loader
  - Embedding index
  - LLM call wrapper (qwen3:4b via Ollama)
  - Answer evaluator
"""

import json
import os
import re
import time
import numpy as np
from pathlib import Path
import ollama
from dotenv import load_dotenv

BASE        = Path(__file__).parent.parent
CORPUS_PATH = BASE / "corpus-20260920T071304Z-1-001/corpus/corpus.jsonl"
EMB_DIR     = BASE / "embeddings"
EMB_MATRIX  = EMB_DIR / "embeddings.npy"
EMB_IDS     = EMB_DIR / "doc_ids.json"
EMB_TITLES  = EMB_DIR / "doc_titles.json"

load_dotenv(BASE / ".env")

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL   = "qwen3:4b"


# ── Corpus ────────────────────────────────────────────────────

class Corpus:
    """Lazy-loaded corpus with O(1) doc_id lookup."""
    def __init__(self):
        self._docs = None

    def _load(self):
        self._docs = {}
        with open(CORPUS_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    self._docs[d["doc_id"]] = d

    @property
    def docs(self):
        if self._docs is None:
            self._load()
        return self._docs

    def get(self, doc_id: str) -> dict | None:
        return self.docs.get(doc_id)

    def get_text(self, doc_id: str, max_chars: int = 2000) -> str:
        d = self.get(doc_id)
        return d["text"][:max_chars] if d else ""

    def get_title(self, doc_id: str) -> str:
        d = self.get(doc_id)
        return d["title"] if d else doc_id


# ── Embedding index ───────────────────────────────────────────

class EmbeddingIndex:
    """
    Loads the pre-built nomic-embed-text embedding matrix.
    Provides cosine similarity search.
    """
    def __init__(self):
        if not EMB_MATRIX.exists():
            raise FileNotFoundError(
                "embeddings/embeddings.npy not found — run embed_documents.py first"
            )
        self.matrix = np.load(EMB_MATRIX).astype(np.float32)
        self.doc_ids = json.loads(EMB_IDS.read_text())
        # Pre-normalise rows for fast cosine via dot product
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        self.normed = self.matrix / (norms + 1e-9)

    def embed_query(self, text: str) -> np.ndarray:
        result = ollama.embed(model=EMBED_MODEL, input=[text])
        vec = np.array(result.embeddings[0], dtype=np.float32)
        return vec / (np.linalg.norm(vec) + 1e-9)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Returns [(doc_id, score), ...] sorted by cosine similarity."""
        q = self.embed_query(query)
        sims = self.normed @ q
        idx  = np.argsort(sims)[::-1][:top_k]
        return [(self.doc_ids[i], float(sims[i])) for i in idx]


# ── LLM wrapper ───────────────────────────────────────────────

def llm_call(prompt: str, max_tokens: int = 300,
             system: str = None) -> tuple[str, int]:
    """
    Call qwen3:4b via Ollama chat mode.
    Returns (answer_text, approx_token_count).
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        options={"temperature": 0, "num_predict": max_tokens, "num_ctx": 4096},
        think=False,
    )
    text = response.message.content or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    tokens = response.eval_count + response.prompt_eval_count
    return text, tokens


def llm_complete(prompt: str, max_tokens: int = 250) -> tuple[str, int]:
    """
    Completion-mode call via ollama.generate.
    Returns (full_response_text, approx_token_count).
    """
    response = ollama.generate(
        model=LLM_MODEL,
        prompt=prompt,
        options={"temperature": 0, "num_predict": max_tokens, "num_ctx": 4096},
        think=False,
    )
    text = response.response or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    tokens = response.eval_count + response.prompt_eval_count
    return text, tokens


def _strip_country_suffix(val: str) -> str:
    """Strip trailing 'from Country (CODE)' annotations from medalist names."""
    val = re.sub(r"\s+from\s+\w[\w\s]*(\([A-Z]{2,3}\))?$", "", val, flags=re.IGNORECASE).strip()
    val = re.sub(r"\s+\([A-Z]{2,3}\)\s*$", "", val).strip()  # trailing (TUR) etc.
    return val


def extract_final_answer(text: str) -> str:
    """
    qwen3:4b reasons before answering. Extract the concise final answer.
    Searches ALL lines — model often names the answer mid-reasoning.

    Priority (checked in this order):
      1. Infobox "gold: X" — most reliable; model quotes the infobox verbatim
      2. Explicit "Answer: X" / "the answer is X" statements
      3. "gold medalist is/was X", "X won the gold medal" subject extraction
      4. Last short line (<= 50 chars) that isn't a reasoning opener
      5. Last non-empty line
    """
    text = text.strip()
    if not text:
        return ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""

    def clean(s):
        s = re.sub(r"\*+", "", s).strip().rstrip(".,;:")
        return s.strip('"\'')  # strip surrounding quotes

    # 1. Infobox "gold: X" — highest reliability, appears when model quotes doc
    for line in lines:  # forward — first match is most likely the right doc
        m = re.search(r"\bgold:\s*([^\n,|]+)", line, re.IGNORECASE)
        if m:
            val = clean(m.group(1))
            if val and len(val) <= 60 and not val.lower().startswith("medal"):
                return val

    # 2. Explicit answer statements — last occurrence wins
    for line in reversed(lines):
        lower = line.lower()
        for prefix in ("answer:", "the answer is:", "the answer is ",
                       "so the answer is ", "therefore the answer is ",
                       "thus the answer is "):
            idx = lower.find(prefix)
            if idx >= 0:
                rest = clean(line[idx + len(prefix):])
                if rest and len(rest) <= 80:
                    return rest

    # 3. "gold medalist here is X" / "gold medalist was X" / "X won the gold"
    _question_words = {"who", "what", "when", "where", "which", "how", "why"}
    for line in reversed(lines):
        # Extract subject from "X won the gold medal" (subject precedes "won")
        m = re.search(r"([A-ZÀ-ÿ][a-zA-ZÀ-ÿ\s\-\.\']+)\s+won\s+the\s+gold", line)
        if m:
            val = clean(m.group(1).strip())
            val = _strip_country_suffix(val)
            if val and len(val) <= 60 and val.lower() not in _question_words:
                return val
        # Extract from "gold medalist is/was/here is X"
        m = re.search(r"gold\s+medalist(?:\s+here)?\s+(?:is|was)\s+([^\n,]+)", line, re.IGNORECASE)
        if m:
            val = _strip_country_suffix(clean(m.group(1).strip()))
            if val and len(val) <= 60:
                return val

    # 4. Last short line that is not an obvious reasoning opener or bullet
    reasoning_starters = {"okay", "ok", "so", "now", "first", "next", "let",
                          "looking", "wait", "hmm", "thus", "then", "since",
                          "because", "doc", "starting", "checking", "the", "i",
                          "-", "•", "*"}
    for line in reversed(lines):
        stripped = clean(line)
        if not stripped:
            continue
        first_word = stripped.split()[0].lower()
        if first_word in reasoning_starters:
            continue
        # Skip lines that are clearly doc-listing bullets "- Doc N: ..."
        if re.match(r"^[-•]\s*(doc\s+\d|[0-9])", stripped, re.IGNORECASE):
            continue
        if len(stripped) <= 50:
            return stripped

    return clean(lines[-1])


# ── Answer evaluation ─────────────────────────────────────────

def normalise(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def score_answer(predicted: str, gold_list: list[str]) -> dict:
    """
    Returns:
      exact_match   – 1 if normalised prediction matches any gold answer
      contains_gold – 1 if prediction contains any gold answer as substring
      gold_in_pred  – 1 if any gold token appears in prediction
    """
    pred_norm = normalise(predicted)
    results = {
        "exact_match"  : 0,
        "contains_gold": 0,
        "gold_in_pred" : 0,
    }
    for gold in gold_list:
        gold_norm = normalise(gold)
        if pred_norm == gold_norm:
            results["exact_match"] = 1
        if gold_norm in pred_norm:
            results["contains_gold"] = 1
        # Token-level: any gold word in prediction
        gold_tokens = set(gold_norm.split())
        pred_tokens = set(pred_norm.split())
        if gold_tokens & pred_tokens:
            results["gold_in_pred"] = 1
    return results


# ── Shared result schema ──────────────────────────────────────

def make_result(
    pipeline: str,
    qid: str,
    question: str,
    qtype: str,
    answer: str,
    gold: list[str],
    retrieved_ids: list[str],
    tokens: int,
    elapsed: float,
    extra: dict = None,
) -> dict:
    scores = score_answer(answer, gold)
    return {
        "pipeline"        : pipeline,
        "qid"             : qid,
        "question"        : question,
        "qtype"           : qtype,
        "predicted_answer": answer,
        "gold_answer"     : gold,
        "retrieved_doc_ids": retrieved_ids,
        "tokens_used"     : tokens,
        "elapsed_sec"     : round(elapsed, 2),
        **scores,
        **(extra or {}),
    }
