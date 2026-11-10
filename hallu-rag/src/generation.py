"""
Generation module for answer generation with citations.
Phase 4: Generate answers using retrieved chunks with inline citations.
"""
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm

from .config import get_config
from .utils import setup_logging, clean_text


logger = setup_logging()


class Generator:
    """Answer generator using local LLM with citation support."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize generator.

        Args:
            model_name: Hugging Face model name (from config if None)
            max_new_tokens: Max tokens to generate (from config if None)
            temperature: Sampling temperature (from config if None)
            top_p: Top-p sampling (from config if None)
            system_prompt: System prompt (from config if None)
        """
        cfg = get_config()
        gen_config = cfg.generation

        self.model_name = model_name or gen_config.get("local_model", "microsoft/Phi-3-mini-4k-instruct")
        self.max_new_tokens = max_new_tokens or gen_config.get("max_new_tokens", 512)
        self.temperature = temperature or gen_config.get("temperature", 0.3)
        self.top_p = top_p or gen_config.get("top_p", 0.9)
        self.system_prompt = system_prompt or gen_config.get("system_prompt", "")

        self.device = cfg.get_device()
        logger.info(f"Loading generation model: {self.model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        if self.device != "cuda":
            self.model = self.model.to(self.device)

        self.model.eval()

        # Set up generation pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks as context with citation markers.

        Args:
            retrieved_chunks: List of retrieved chunk dictionaries

        Returns:
            Formatted context string
        """
        context_parts = []
        for chunk in retrieved_chunks:
            citation_id = chunk.get("citation_id", "C?")
            text = chunk.get("text", "")
            context_parts.append(f"[{citation_id}] {text}")

        return "\n\n".join(context_parts)

    def build_prompt(self, question: str, context: str) -> str:
        """
        Build the full prompt for generation.

        Args:
            question: User question
            context: Formatted context with citations

        Returns:
            Full prompt string
        """
        # Phi-3 format
        if "phi-3" in self.model_name.lower():
            prompt = f"<|system|>\n{self.system_prompt}<|end|>\n<|user|>\nContext:\n{context}\n\nQuestion: {question}<|end|>\n<|assistant|>\n"
        # Generic chat format
        else:
            prompt = f"{self.system_prompt}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        return prompt

    def generate(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate answer for a question using retrieved chunks.

        Args:
            question: User question
            retrieved_chunks: List of retrieved chunks with citation_id

        Returns:
            Dictionary with answer, citations, and metadata
        """
        if not retrieved_chunks:
            return {
                "answer": "I don't have enough information to answer this question.",
                "citations": [],
                "context_used": [],
            }

        # Format context
        context = self.format_context(retrieved_chunks)

        # Build prompt
        prompt = self.build_prompt(question, context)

        # Generate
        logger.debug(f"Generating answer for: {question[:100]}...")

        outputs = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.temperature > 0,
            return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        answer = outputs[0]["generated_text"].strip()

        # Extract citations from answer
        citations = self._extract_citations(answer, retrieved_chunks)

        return {
            "answer": answer,
            "citations": citations,
            "context_used": [c["chunk_id"] for c in retrieved_chunks],
            "prompt_tokens": len(self.tokenizer.encode(prompt)),
            "completion_tokens": len(self.tokenizer.encode(answer)),
        }

    def _extract_citations(
        self,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Extract citation markers from generated answer.

        Args:
            answer: Generated answer text
            retrieved_chunks: Retrieved chunks

        Returns:
            List of citation dictionaries
        """
        import re

        # Find all citation patterns like [C1], [C2], etc.
        citation_pattern = r"\[C(\d+)\]"
        matches = re.findall(citation_pattern, answer)

        # Map citation IDs to chunk info
        citation_map = {c["citation_id"]: c for c in retrieved_chunks}

        citations = []
        for match in matches:
            cit_id = f"C{match}"
            if cit_id in citation_map:
                chunk = citation_map[cit_id]
                citations.append({
                    "citation_id": cit_id,
                    "chunk_id": chunk["chunk_id"],
                    "paper_id": chunk.get("paper_id", ""),
                    "paper_title": chunk.get("paper_title", ""),
                    "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                })

        # Deduplicate
        seen = set()
        unique_citations = []
        for c in citations:
            if c["citation_id"] not in seen:
                seen.add(c["citation_id"])
                unique_citations.append(c)

        return unique_citations

    def generate_batch(
        self,
        questions: List[str],
        retrieved_chunks_list: List[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        Generate answers for multiple questions.

        Args:
            questions: List of questions
            retrieved_chunks_list: List of retrieved chunks for each question

        Returns:
            List of generation results
        """
        results = []
        for question, chunks in tqdm(zip(questions, retrieved_chunks_list), total=len(questions), desc="Generating answers"):
            result = self.generate(question, chunks)
            results.append(result)
        return results


def create_generator(
    model_name: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> Generator:
    """Factory function to create a generator."""
    return Generator(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )


if __name__ == "__main__":
    # Smoke test - just verify imports
    cfg = get_config()
    print("Generation module loaded successfully")
    print(f"Model: {cfg.generation.get('local_model')}")
    print(f"Device: {cfg.get_device()}")