# Runs benchmark.json end-to-end through your actual retrieve -> rerank -> generate
# pipeline and reports both retrieval and generation metrics. Run this after every
# change (new chunking logic, new prompt, new model, new reranker...) and compare
# against the previous run to know if you made things better or worse.
#
# Usage:
#   python run_evaluation.py --user_id alice --benchmark benchmark.json --backend ollama

import argparse
import json
import os
from datetime import datetime, timezone

import chromadb
from sentence_transformers import SentenceTransformer
import ingest
import config
import rag
import llm_providers
import eval_metrics as em

RESULTS_DIR = "eval_results"


def load_benchmark(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cases = data["test_cases"]
    return cases


def build_context_text(chunks: list[dict]) -> str:
    #Mirrors the text portion of rag.build_prompt, kept separate so the judge sees
    #exactly what the generator saw without needing to touch rag.py.
    parts = []
    for c in chunks:
        meta = c["metadata"]
        parts.append(f"--- Source ({meta['source_file']}, page {meta['page']}, type: {meta['type']}) ---\n{c['content']}")
    return "\n\n".join(parts)


def generate_answer_no_rerank(query: str, reranked_chunks: list[dict], backend: str) -> str:
    #Mirrors rag.generate_answer exactly, EXCEPT it skips generate_answer's internal
    #rerank_chunks(...) call, because run_one_case already reranked these chunks once.
    #Calling rag.generate_answer directly here would rerank the same ~40 pairs a second
    #time -- on CPU, with a heavier multilingual cross-encoder, that doubles a multi-minute
    #cost for zero benefit (identical input, identical output).
    backend = (backend or config.LLM_BACKEND).lower()
    chunks = reranked_chunks
    include_images = backend != "groq"
    if backend == "ollama":
        chunks = rag.fit_chunks_to_context(chunks, config.CONTEXT_WINDOW, reserved_for_answer=1200)
    prompt, images_b64 = rag.build_prompt(query, chunks, history_text="", include_images=include_images)
    provider = llm_providers.get_llm_provider(backend)
    try:
        return provider.generate(prompt, images=images_b64 if images_b64 else None)
    except Exception as e:
        return f"Sorry, I encountered an error while generating the answer: {e}"


def run_one_case(case: dict, collection, embed_model, backend: str, judge_provider, top_k: int,user_id: str) -> dict:
    import time
    question = case["question"]
    translated_sources = translate_relevant_sources(case.get("relevant_sources", []), user_id)
    relevant_keys = em.relevant_to_source_keys(translated_sources)

    # --- Retrieval stage (mirrors app.py / rag.py) ---
    t0 = time.time()
    raw_chunks = rag.retrieve_chunks(question, collection, embed_model, top_k=top_k)
    print(f"    retrieve_chunks: {time.time()-t0:.1f}s ({len(raw_chunks)} chunks)", flush=True)

    t0 = time.time()
    raw_keys = em.chunks_to_source_keys(raw_chunks)
    retrieval_pre_rerank = em.retrieval_scorecard(raw_keys, relevant_keys, pool_k=top_k, metric_k=5)

    reranked_chunks = rag.rerank_chunks(question, raw_chunks, top_n=config.rerank_top_n)
    print(f"    rerank_chunks:   {time.time()-t0:.1f}s ({len(reranked_chunks)} chunks)", flush=True)
    reranked_keys = em.chunks_to_source_keys(reranked_chunks)
    retrieval_post_rerank = em.retrieval_scorecard(reranked_keys, relevant_keys, pool_k=config.rerank_top_n, metric_k=5)

    # --- Generation stage (reuses your real pipeline, no memory for a clean standalone run) ---
    t0 = time.time()
    answer = generate_answer_no_rerank(question, reranked_chunks, backend)
    print(f"    generate_answer: {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    context_text = build_context_text(reranked_chunks)
    generation = em.generation_scorecard(
        question=question,
        context_text=context_text,
        answer=answer,
        reference_answer=case.get("expected_answer", ""),
        judge_provider=judge_provider,
        embed_model=embed_model,
    )
    print(f"    judge calls:     {time.time()-t0:.1f}s", flush=True)

    return {
        "id": case["id"],
        "question": question,
        "category": case.get("category"),
        "difficulty": case.get("difficulty"),
        "answer": answer,
        "retrieved_sources_raw": raw_keys,
        "retrieved_sources_reranked": reranked_keys,
        "retrieval_pre_rerank": retrieval_pre_rerank,
        "retrieval_post_rerank": retrieval_post_rerank,
        "generation": generation,
    }

def compare_to_previous(current_agg: dict, threshold: float = 0.05) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    latest_path = os.path.join(RESULTS_DIR, "latest.json")
    if not os.path.exists(latest_path):
        print("\n(No previous run found -- this becomes the baseline.)")
        return

    with open(latest_path, encoding="utf-8") as f:
        previous = json.load(f)
    prev_agg = previous.get("aggregated", {})

    print("\n--- Change vs previous run ---")
    for key, current_value in current_agg.items():
        prev_value = prev_agg.get(key)
        if current_value is None or prev_value is None:
            continue
        delta = current_value - prev_value
        flag = ""
        if delta <= -threshold:
            flag = "  <-- REGRESSION"
        elif delta >= threshold:
            flag = "  <-- IMPROVED"
        print(f"  {key:28s} {prev_value:.3f} -> {current_value:.3f} ({delta:+.3f}){flag}")

_page_map_cache: dict[str, dict] = {}

def get_printed_to_physical_map(user_id: str, source_file: str) -> dict:
    """Cached per fichier -- reconstruire la map nécessite de rouvrir le PDF, donc une seule fois."""
    cache_key = f"{user_id}::{source_file}"
    if cache_key in _page_map_cache:
        return _page_map_cache[cache_key]

    ext = os.path.splitext(source_file)[1].lower()
    if ext != ".pdf":
        _page_map_cache[cache_key] = {}  # seuls les PDF ont ce décalage de front-matter
        return {}

    file_path = os.path.join(config.get_user_data_dir(user_id), source_file)
    if not os.path.exists(file_path):
        _page_map_cache[cache_key] = {}
        return {}

    page_map = ingest.build_page_number_map(file_path)
    reverse = ingest.build_printed_to_physical_map(page_map)
    _page_map_cache[cache_key] = reverse
    return reverse


def translate_relevant_sources(relevant_sources: list[dict], user_id: str) -> list[dict]:
    """Convertit les pages imprimées du benchmark vers l'index physique Docling."""
    translated = []
    for r in relevant_sources:
        printed_page = str(r["page"])
        reverse_map = get_printed_to_physical_map(user_id, r["source_file"])
        physical_page = reverse_map.get(printed_page, r["page"])  # fallback: valeur telle quelle si pas de mapping
        translated.append({"source_file": r["source_file"], "page": physical_page})
    return translated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", required=True, help="Whose collection the benchmark was built from")
    parser.add_argument("--benchmark", default="benchmark.json")
    parser.add_argument("--backend", default=None, help="Assistant backend to evaluate (ollama/groq). Default: config.LLM_BACKEND")
    parser.add_argument("--judge_backend", default=None, help="Backend used as the grader. Default: same as --backend")
    parser.add_argument("--top_k", type=int, default=None, help="Default: config.TOP_K")
    args = parser.parse_args()

    backend = args.backend or config.LLM_BACKEND
    judge_backend = args.judge_backend or backend
    top_k = args.top_k or config.TOP_K

    print(f"Loading embedding model and connecting to ChromaDB (backend={backend}, judge={judge_backend})...")
    embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection = client.get_or_create_collection(config.get_user_collection_name(args.user_id))
    judge_provider = llm_providers.get_llm_provider(judge_backend)

    cases = load_benchmark(args.benchmark)
    print(f"Running {len(cases)} test cases...\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    partial_path = os.path.join(RESULTS_DIR, f"run_{timestamp}_partial.json")
 
    def save_progress(results):
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": timestamp, "backend": backend, "judge_backend": judge_backend,
                "status": "in_progress", "completed": len(results), "total": len(cases),
                "per_case": results,
            }, f, ensure_ascii=False, indent=2)
 
    per_case_results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']}: {case['question'][:70]}", flush=True)
        try:
            result = run_one_case(case, collection, embed_model, backend, judge_provider, top_k, args.user_id)
        except Exception as e:
            print(f"  !! {case['id']} FAILED: {e} -- recorded as error, continuing", flush=True)
            result = {"id": case["id"], "question": case["question"],
                      "category": case.get("category"), "error": str(e)}
        per_case_results.append(result)
        save_progress(per_case_results)
 
    ok_results = [r for r in per_case_results if "generation" in r]
    failed_ids = [r["id"] for r in per_case_results if "generation" not in r]
    if failed_ids:
        print(f"\n(!) {len(failed_ids)} case(s) failed and were excluded from aggregation: {failed_ids}")
 
    retrieval_cards = [r["retrieval_post_rerank"] for r in ok_results]
    generation_cards = [r["generation"] for r in ok_results]
    aggregated = {}
    aggregated.update({f"retrieval_{k}": v for k, v in em.aggregate(retrieval_cards).items()})
    aggregated.update({f"generation_{k}": v for k, v in em.aggregate(generation_cards).items()})

    correctness_scores = [c.get("correctness") for c in generation_cards if isinstance(c.get("correctness"), (int, float))]
    if correctness_scores:
        aggregated["generation_accuracy_strict"] = sum(1 for s in correctness_scores if s == 1.0) / len(correctness_scores)
 
    # Hallucination rate: the inverse framing of faithfulness, since this is usually the
    # number people actually want to report ("X% of answers contained unsupported claims").
    if aggregated.get("generation_faithful") is not None:
        aggregated["generation_hallucination_rate"] = 1 - aggregated["generation_faithful"]

    if correctness_scores:
        n_correct = sum(1 for s in correctness_scores if s == 1.0)
        n_total = len(correctness_scores)
        print(f"\n=== Overall accuracy: {n_correct}/{n_total} correct ({n_correct/n_total:.0%}) ===")
 
    print("\n=== Aggregated results ===")
    for k, v in sorted(aggregated.items()):
        print(f"  {k:28s} {v:.3f}" if v is not None else f"  {k:28s} n/a")
 
    categories = sorted({r.get("category") for r in ok_results if r.get("category")})
    if len(categories) > 1:
        print("\n=== By category ===")
        for cat in categories:
            cat_cases = [r["generation"] for r in ok_results if r.get("category") == cat]
            cat_scores = [c.get("correctness") for c in cat_cases if isinstance(c.get("correctness"), (int, float))]
            cat_faithful = [c.get("faithful") for c in cat_cases if isinstance(c.get("faithful"), (int, float))]
            acc = sum(1 for s in cat_scores if s == 1.0) / len(cat_scores) if cat_scores else None
            faith = sum(cat_faithful) / len(cat_faithful) if cat_faithful else None
            n = len(cat_cases)
            acc_str = f"{acc:.2f}" if acc is not None else "n/a"
            faith_str = f"{faith:.2f}" if faith is not None else "n/a"
            print(f"  {cat:24s} n={n:<3d} accuracy={acc_str:6s} faithful={faith_str}")
 
    compare_to_previous(aggregated)
 
    run_record = {
        "timestamp": timestamp,
        "backend": backend,
        "judge_backend": judge_backend,
        "failed_cases": failed_ids,
        "aggregated": aggregated,
        "per_case": per_case_results,
    }
    with open(os.path.join(RESULTS_DIR, f"run_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(run_record, f, ensure_ascii=False, indent=2)
    with open(os.path.join(RESULTS_DIR, "latest.json"), "w", encoding="utf-8") as f:
        json.dump(run_record, f, ensure_ascii=False, indent=2)
 
    print(f"\nSaved full results to {RESULTS_DIR}/run_{timestamp}.json")
 
 
if __name__ == "__main__":
    main()