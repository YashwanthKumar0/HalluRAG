import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

from src.config import load_config
from src.indexing import load_index, IndexBundle
from src.retrieval import Retriever
from src.generation import generate, build_prompt, parse_citations
from src.faithfulness import NLIScorer, verify_answer, extract_claims


# Module-level caches for models
_embedder_cache: Dict[str, Any] = {}
_cross_encoder_cache: Dict[str, Any] = {}
_nli_scorer_cache: Dict[str, NLIScorer] = {}


def get_embedder(model_name: str, device: str):
    """Get or create cached embedder."""
    key = f"{model_name}:{device}"
    if key not in _embedder_cache:
        from sentence_transformers import SentenceTransformer
        _embedder_cache[key] = SentenceTransformer(model_name, device=device)
    return _embedder_cache[key]


def get_cross_encoder(model_name: str):
    """Get or create cached cross-encoder."""
    if model_name not in _cross_encoder_cache:
        from sentence_transformers import CrossEncoder
        _cross_encoder_cache[model_name] = CrossEncoder(model_name)
    return _cross_encoder_cache[model_name]


def get_nli_scorer(model_name: str) -> NLIScorer:
    """Get or create cached NLI scorer."""
    if model_name not in _nli_scorer_cache:
        _nli_scorer_cache[model_name] = NLIScorer(model_name)
    return _nli_scorer_cache[model_name]


class HalluRAG:
    """
    End-to-end HalluRAG pipeline.

    Args:
        cfg: Configuration dictionary
        strategy: Chunking strategy ("fixed" or "semantic")
        mode: Retrieval mode ("dense", "bm25", "hybrid")
        use_rerank: Whether to use cross-encoder reranking
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        strategy: str = "semantic",
        mode: str = "hybrid",
        use_rerank: bool = True
    ):
        self.cfg = cfg
        self.strategy = strategy
        self.mode = mode
        self.use_rerank = use_rerank

        # Load index bundle
        self.bundle = load_index(strategy, cfg)

        # Build retriever
        self.retriever = Retriever(self.bundle, cfg)

        # Faithfulness config
        self.faithfulness_cfg = cfg.get("faithfulness", {})
        nli_model = self.faithfulness_cfg.get("nli_model", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
        self.nli_scorer = get_nli_scorer(nli_model)

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and return full pipeline result.

        Returns:
            dict with keys: question, answer, chunks, citations, verification, timings
        """
        timings = {}
        start_total = time.time()

        # 1. Retrieval
        t0 = time.time()
        chunks = self.retriever.retrieve(
            question,
            mode=self.mode,
            use_rerank=self.use_rerank,
            top_k=self.cfg.get("retrieval", {}).get("top_k", 5)
        )
        timings["retrieval"] = time.time() - t0

        # 2. Generation
        t0 = time.time()
        gen_result = generate(question, chunks, self.cfg)
        timings["generation"] = time.time() - t0

        answer = gen_result["answer"]
        citations = gen_result["citations"]

        # 3. Faithfulness verification
        t0 = time.time()
        if gen_result.get("status") == "llm_error":
            verification = {
                "claims": [],
                "faithfulness": None,
                "n_supported": 0,
                "n_unverified": 0,
                "n_unsupported": 0,
                "abstained": False,
                "overall_flag": "error",
            }
        else:
            verification = verify_answer(
                answer,
                citations,
                chunks,
                self.nli_scorer,
                self.faithfulness_cfg,
                relevance_scorer=self.retriever.cross_encoder,
            )
        timings["faithfulness"] = time.time() - t0

        timings["total"] = time.time() - start_total

        return {
            "question": question,
            "answer": answer,
            "chunks": chunks,
            "citations": citations,
            "verification": verification,
            "status": gen_result.get("status", "ok"),
            "error": gen_result.get("error"),
            "timings": timings
        }


def run_cli():
    """Run pipeline from command line."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline \"your question here\"")
        print("Optional args: --strategy fixed|semantic --mode dense|bm25|hybrid --rerank true|false")
        sys.exit(1)

    # Parse args
    question = sys.argv[1]
    strategy = "semantic"
    mode = "hybrid"
    use_rerank = True

    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--strategy" and i + 1 < len(sys.argv):
            strategy = sys.argv[i + 1]
        elif arg == "--mode" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
        elif arg == "--rerank" and i + 1 < len(sys.argv):
            use_rerank = sys.argv[i + 1].lower() == "true"

    cfg = load_config()
    pipeline = HalluRAG(cfg, strategy=strategy, mode=mode, use_rerank=use_rerank)

    print(f"\n=== HalluRAG Pipeline ===")
    print(f"Strategy: {strategy}, Mode: {mode}, Rerank: {use_rerank}")
    print(f"Question: {question}\n")

    result = pipeline.ask(question)

    print(f"Answer: {result['answer']}\n")

    print("Cited chunks:")
    for chunk in result['chunks']:
        print(f"  [{chunk['rank']}] {chunk['chunk_id']} (score: {chunk['score']:.4f})")
        print(f"      Title: {chunk['title'][:80]}")
        print(f"      Section: {chunk['section']}")

    print("\nVerification:")
    ver = result['verification']
    print(f"  Overall flag: {ver['overall_flag'].upper()}")
    print(f"  Faithfulness: {ver['faithfulness']}")
    print(f"  Supported: {ver['n_supported']}, Unverified: {ver['n_unverified']}, Unsupported: {ver['n_unsupported']}")
    print(f"  Abstained: {ver['abstained']}")

    if ver['claims']:
        print("\nPer-claim verdicts:")
        for i, claim in enumerate(ver['claims'], 1):
            print(f"  {i}. {claim['label'].upper()} (entail={claim['entail']:.3f}, contradict={claim['contradict']:.3f})")
            print(f"     Claim: {claim['claim'][:100]}...")
            print(f"     Best evidence: {claim['best_evidence'][:100]}...")
            print(f"     From chunk: {claim['best_chunk_id']}")

    print(f"\nTimings (seconds):")
    for stage, t in result['timings'].items():
        print(f"  {stage}: {t:.3f}s")


if __name__ == "__main__":
    run_cli()