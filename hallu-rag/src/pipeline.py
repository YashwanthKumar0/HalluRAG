"""
Main RAG pipeline tying retrieval, generation, and faithfulness verification.
"""
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from .config import get_config
from .retrieval import Retriever, create_retriever
from .generation import Generator, create_generator
from .faithfulness import FaithfulnessVerifier, create_verifier
from .utils import setup_logging


logger = setup_logging()


@dataclass
class RAGResult:
    """Complete RAG pipeline result."""
    question: str
    answer: str
    citations: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    verification: Dict[str, Any]
    config: Dict[str, Any]  # Configuration used

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RAGPipeline:
    """Complete RAG pipeline with retrieval, generation, and verification."""

    def __init__(
        self,
        strategy: str = "fixed",
        retrieval_type: str = "dense",
        use_rerank: bool = False,
        indexes: Optional[Dict[str, Any]] = None,
        chunks_dir: Optional[Path] = None,
        index_dir: Optional[Path] = None,
    ):
        """
        Initialize RAG pipeline.

        Args:
            strategy: Chunking strategy ("fixed" or "semantic")
            retrieval_type: Retrieval type ("dense" or "hybrid")
            use_rerank: Whether to use cross-encoder reranking
            indexes: Pre-loaded indexes
            chunks_dir: Chunks directory
            index_dir: Index directory
        """
        self.strategy = strategy
        self.retrieval_type = retrieval_type
        self.use_rerank = use_rerank

        # Initialize components
        self.retriever = create_retriever(
            strategy=strategy,
            retrieval_type=retrieval_type,
            use_rerank=use_rerank,
            indexes=indexes,
            chunks_dir=chunks_dir,
            index_dir=index_dir,
        )

        self.generator = create_generator()
        self.verifier = create_verifier()

        # Config for tracking
        self.config = {
            "strategy": strategy,
            "retrieval_type": retrieval_type,
            "use_rerank": use_rerank,
        }

    def run(self, question: str, top_k: Optional[int] = None) -> RAGResult:
        """
        Run the complete RAG pipeline for a question.

        Args:
            question: User question
            top_k: Number of chunks to retrieve (from config if None)

        Returns:
            Complete RAGResult
        """
        logger.info(f"Processing question: {question[:100]}...")

        # Step 1: Retrieve
        retrieved_chunks = self.retriever.retrieve(question, top_k)
        logger.debug(f"Retrieved {len(retrieved_chunks)} chunks")

        # Step 2: Generate
        generation_result = self.generator.generate(question, retrieved_chunks)
        answer = generation_result["answer"]
        citations = generation_result["citations"]
        logger.debug(f"Generated answer ({len(answer)} chars)")

        # Step 3: Verify
        verification = self.verifier.verify_answer(answer, retrieved_chunks)
        logger.debug(f"Verification: {verification['overall_faithfulness']}")

        return RAGResult(
            question=question,
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            verification=verification,
            config=self.config.copy(),
        )

    def run_batch(self, questions: List[str], top_k: Optional[int] = None) -> List[RAGResult]:
        """
        Run pipeline for multiple questions.

        Args:
            questions: List of questions
            top_k: Number of chunks to retrieve

        Returns:
            List of RAGResults
        """
        results = []
        for question in questions:
            result = self.run(question, top_k)
            results.append(result)
        return results


def create_pipeline(
    strategy: str = "fixed",
    retrieval_type: str = "dense",
    use_rerank: bool = False,
    indexes: Optional[Dict[str, Any]] = None,
    chunks_dir: Optional[Path] = None,
    index_dir: Optional[Path] = None,
) -> RAGPipeline:
    """Factory function to create a RAG pipeline."""
    return RAGPipeline(
        strategy=strategy,
        retrieval_type=retrieval_type,
        use_rerank=use_rerank,
        indexes=indexes,
        chunks_dir=chunks_dir,
        index_dir=index_dir,
    )


if __name__ == "__main__":
    # Smoke test
    cfg = get_config()
    print("Pipeline module loaded successfully")
    print("Available configs:")
    print(f"  Strategies: fixed, semantic")
    print(f"  Retrieval types: dense, hybrid")
    print(f"  Rerank: true, false")