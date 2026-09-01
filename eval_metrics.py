# Metric functions for the two-part evaluation:
#   Part 1 - retrieval metrics: did we fetch the right chunks?
#   Part 2 - generation metrics: is the answer grounded in what we fetched (no hallucination)?

import json
import logging
import re
import numpy as np

logger = logging.getLogger(__name__)


def _source_key(source_file: str, page) -> str:
    return f"{source_file}::{page}"


def chunks_to_source_keys(chunks: list[dict]) -> list[str]:
    keys = []
    for c in chunks:
        meta = c["metadata"]
        start_page = meta["page"]
        end_page = meta.get("page_end", start_page) or start_page
        for page in range(start_page, end_page + 1):
            keys.append(_source_key(meta["source_file"], page))
    return keys


def relevant_to_source_keys(relevant_sources: list[dict]) -> set:
    return {_source_key(r["source_file"], r["page"]) for r in relevant_sources}


#Part 1: Retrieval metrics

def precision_at_k(retrieved_keys: list[str], relevant_keys: set, k: int) -> float:
    effective_k = min(k, len(retrieved_keys))
    if effective_k <= 0:
        return 0.0
    top_k = set(retrieved_keys[:effective_k])
    return len(top_k & relevant_keys) / effective_k


def recall_at_k(retrieved_keys: list[str], relevant_keys: set, k: int) -> float:
    if not relevant_keys:
        return None
    top_k = set(retrieved_keys[:k])
    return len(top_k & relevant_keys) / len(relevant_keys)


def f1_at_k(retrieved_keys: list[str], relevant_keys: set, k: int) -> float:
    p = precision_at_k(retrieved_keys, relevant_keys, k)
    r = recall_at_k(retrieved_keys, relevant_keys, k)
    if r is None or (p + r) == 0:
        return 0.0
    return 2 * p * r / (p + r)


def reciprocal_rank(retrieved_keys: list[str], relevant_keys: set) -> float:
    for i, key in enumerate(retrieved_keys, start=1):
        if key in relevant_keys:
            return 1 / i
    return 0.0


def hit_rate_at_k(retrieved_keys: list[str], relevant_keys: set, k: int) -> int:
    if not relevant_keys:
        return None
    return 1 if set(retrieved_keys[:k]) & relevant_keys else 0


def retrieval_scorecard(retrieved_keys: list[str], relevant_keys: set, pool_k: int, metric_k: int = 5) -> dict:
    """
    All retrieval metrics for one test case, bundled together.

    pool_k: size of the candidate pool actually retrieved at this stage (e.g. config.TOP_K
            for raw retrieval, or rerank_top_n for the post-rerank stage). Used for
            recall/hit-rate: "was the right chunk in what we fetched at all?" -- this
            SHOULD improve as pool_k grows, and isn't comparable across different pool_k.

    metric_k: a FIXED window (default 5) used for precision/F1 so these stay comparable
              across experiments even when pool_k changes between runs (e.g. testing
              top_k=20 vs top_k=40). Effective window is min(metric_k, len(retrieved_keys)).
    """
    return {
        "recall_at_pool": recall_at_k(retrieved_keys, relevant_keys, pool_k),
        "hit_rate_at_pool": hit_rate_at_k(retrieved_keys, relevant_keys, pool_k),
        "reciprocal_rank": reciprocal_rank(retrieved_keys, relevant_keys),
        "precision_at_fixed_k": precision_at_k(retrieved_keys, relevant_keys, metric_k),
        "recall_at_fixed_k": recall_at_k(retrieved_keys, relevant_keys, metric_k),
        "f1_at_fixed_k": f1_at_k(retrieved_keys, relevant_keys, metric_k),
    }


#Part 2: Generation metrics

FAITHFULNESS_PROMPT = """You are a strict fact-checker. Below is a CONTEXT (retrieved
document excerpts) and an ANSWER a system produced from that context.

Judge ONLY whether every factual claim in the ANSWER is supported by the CONTEXT.
It is fine if the answer is incomplete. It is NOT fine if the answer states something
the context does not support (that is a hallucination).

IMPORTANT: If the ANSWER simply declines to answer or states that the information
is not available (a refusal), and makes no other factual claim, it should always be
judged SUPPORTED — there is no unsupported claim to flag in a refusal.

Respond with ONLY one word: SUPPORTED or UNSUPPORTED.

CONTEXT:
\"\"\"{context}\"\"\"

ANSWER:
\"\"\"{answer}\"\"\"
"""

RELEVANCY_PROMPT = """Rate how directly the ANSWER addresses the QUESTION, from 1 to 5.
5 = fully addresses it, 1 = does not address it at all (e.g. off-topic, or refuses without cause).
Respond with ONLY a single digit 1-5.

QUESTION: {question}
ANSWER: {answer}
"""

CORRECTNESS_PROMPT = """Compare the CANDIDATE answer to the EXPERT REFERENCE answer for the
same question. Judge only factual correctness of the candidate relative to the reference,
not writing style or verbosity.

Respond with ONLY one word: CORRECT, PARTIALLY_CORRECT, or INCORRECT.

QUESTION: {question}
EXPERT REFERENCE ANSWER: {reference}
CANDIDATE ANSWER: {candidate}
"""
REFUSAL_PATTERNS = [
    r"les documents (fournis )?ne (contiennent|fournissent|pr[ée]cisent) pas",
    r"le document ne (fournit|pr[ée]cise|mentionne) pas",
    r"the documents? (don't|do not) provide",
    r"the documents? (don't|do not) (contain|specify|mention)",
    r"context does not (provide|contain|specify)",
]
def _is_pure_refusal(answer: str, max_len: int = 220) -> bool:
    text = answer.strip()
    if len(text) > max_len:
        return False
    lowered = text.lower()
    return any(re.search(pat, lowered) for pat in REFUSAL_PATTERNS)

def judge_faithfulness(context_text: str, answer: str, judge_provider) -> dict:
    if _is_pure_refusal(answer):
        return {
            "faithful": 1.0,
            "faithful_raw": "SUPPORTED (auto-detected pure refusal, judge not called)",
        }
    raw = judge_provider.generate(FAITHFULNESS_PROMPT.format(context=context_text, answer=answer))
    verdict = raw.strip().upper()
    score = 1.0 if "UNSUPPORTED" not in verdict and "SUPPORTED" in verdict else 0.0
    return {"faithful": score, "faithful_raw": raw.strip()}


def judge_relevancy(question: str, answer: str, judge_provider) -> dict:
    raw = judge_provider.generate(RELEVANCY_PROMPT.format(question=question, answer=answer))
    digits = "".join(ch for ch in raw if ch.isdigit())
    score = int(digits[0]) / 5 if digits else None
    return {"relevancy": score, "relevancy_raw": raw.strip()}


def judge_correctness(question: str, reference: str, candidate: str, judge_provider) -> dict:
    raw = judge_provider.generate(
        CORRECTNESS_PROMPT.format(question=question, reference=reference, candidate=candidate)
    )
    verdict = raw.strip().upper()
    if "PARTIALLY" in verdict:
        score = 0.5
    elif "INCORRECT" in verdict:
        score = 0.0
    elif "CORRECT" in verdict:
        score = 1.0
    else:
        score = None
    return {"correctness": score, "correctness_raw": raw.strip()}


def answer_similarity(reference_answer: str, answer: str, embed_model) -> float | None:
    if not reference_answer or not answer:
        return None
    emb_ref = embed_model.encode(reference_answer)
    emb_ans = embed_model.encode(answer)
    denom = np.linalg.norm(emb_ref) * np.linalg.norm(emb_ans)
    if denom == 0:
        return 0.0
    return float(np.dot(emb_ref, emb_ans) / denom)


def generation_scorecard(question: str, context_text: str, answer: str,reference_answer: str, judge_provider, embed_model=None) -> dict:
    #All generation metrics for one test case, bundled together.
    result = {}
    result.update(judge_faithfulness(context_text, answer, judge_provider))
    result.update(judge_relevancy(question, answer, judge_provider))
    if reference_answer:
        result.update(judge_correctness(question, reference_answer, answer, judge_provider))
        if embed_model is not None:
            result["answer_similarity"] = answer_similarity(reference_answer, answer, embed_model)
    return result


def aggregate(scorecards: list[dict]) -> dict:
    #Average every numeric field across a list of scorecards, skipping None values.
    if not scorecards:
        return {}
    keys = {k for card in scorecards for k, v in card.items() if isinstance(v, (int, float))}
    out = {}
    for key in keys:
        values = [card[key] for card in scorecards if isinstance(card.get(key), (int, float))]
        out[key] = sum(values) / len(values) if values else None
    return out