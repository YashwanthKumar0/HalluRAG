"""
Hallucination-Aware RAG for Domain-Specific QA
===============================================

A Retrieval-Augmented Generation system with faithfulness verification
for arXiv papers (cs.CL / cs.AI).
"""
from .config import get_config, Config
from .data_prep import Paper, download_papers, process_papers, run_data_prep
from .chunking import Chunk, fixed_chunking, semantic_chunking, run_chunking
from .indexing import DenseIndex, BM25Index, build_all_indexes, load_indexes
from .retrieval import Retriever, create_retriever
from .generation import Generator, create_generator
from .faithfulness import FaithfulnessVerifier, create_verifier
from .pipeline import RAGPipeline, RAGResult, create_pipeline
from .evaluation import (
    EvaluationResult,
    compute_recall_at_k,
    compute_mrr,
    compute_answer_relevance,
    compute_faithfulness_metrics,
    compute_human_agreement,
    evaluate_pipeline,
    run_experiment,
    run_all_experiments,
    save_results,
)
from .utils import (
    setup_logging,
    save_jsonl,
    load_jsonl,
    save_json,
    load_json,
    split_sentences,
    clean_text,
    generate_chunk_id,
    ensure_dir,
)

__version__ = "0.1.0"
__all__ = [
    # Config
    "get_config",
    "Config",
    # Data prep
    "Paper",
    "download_papers",
    "process_papers",
    "run_data_prep",
    # Chunking
    "Chunk",
    "fixed_chunking",
    "semantic_chunking",
    "run_chunking",
    # Indexing
    "DenseIndex",
    "BM25Index",
    "build_all_indexes",
    "load_indexes",
    # Retrieval
    "Retriever",
    "create_retriever",
    # Generation
    "Generator",
    "create_generator",
    # Faithfulness
    "FaithfulnessVerifier",
    "create_verifier",
    # Pipeline
    "RAGPipeline",
    "RAGResult",
    "create_pipeline",
    # Evaluation
    "EvaluationResult",
    "compute_recall_at_k",
    "compute_mrr",
    "compute_answer_relevance",
    "compute_faithfulness_metrics",
    "compute_human_agreement",
    "evaluate_pipeline",
    "run_experiment",
    "run_all_experiments",
    "save_results",
    # Utils
    "setup_logging",
    "save_jsonl",
    "load_jsonl",
    "save_json",
    "load_json",
    "split_sentences",
    "clean_text",
    "generate_chunk_id",
    "ensure_dir",
]