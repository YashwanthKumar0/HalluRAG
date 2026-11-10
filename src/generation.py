import re
import time
import logging
import nltk
from typing import List, Dict, Any, Tuple
from openai import OpenAI

from src.config import load_config

logger = logging.getLogger(__name__)


# Ensure NLTK sentence tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


SYSTEM_PROMPT = """You are a careful research assistant. Answer the question using ONLY the numbered context passages.
Rules:
1. Every sentence in your answer must end with the citation(s) it relies on, e.g. [C2] or [C1][C3].
2. If the context answers the question fully or partly, answer with what it supports. Only if NO passage is relevant, reply exactly: "The provided documents do not contain enough information to answer this question."
3. Do not use outside knowledge. Do not invent numbers, names or results.
4. Be concise: 2-6 sentences."""

def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 characters per token."""
    return len(text) // 4


def strip_reasoning(text: str) -> str:
    """Strip reasoning blocks (e.g., </think>...```, </think>...```, </think>...```) from model output."""
    if not text:
        return ""
    # Strip </think>...``` blocks
    text = re.sub(r'```.*?```', '', text, flags=re.S)
    # Strip </think>...``` blocks
    text = re.sub(r'```.*?```', '', text, flags=re.S)
    # Strip </think>...``` blocks
    text = re.sub(r'```.*?```', '', text, flags=re.S)
    # Strip generic reasoning patterns
    text = re.sub(r'Let me think.*?(?=\n\n|\n$)', '', text, flags=re.S)
    text = re.sub(r'I think.*?(?=\n\n|\n$)', '', text, flags=re.S)
    text = re.sub(r'Reasoning:.*?(?=\n\n|\n$)', '', text, flags=re.S)
    return text.strip()


def build_prompt(question: str, chunks: List[Dict], system_prompt: str = SYSTEM_PROMPT) -> List[Dict[str, str]]:
    """Build chat messages with system prompt and user message containing labeled chunks."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("title", "Unknown")
        section = chunk.get("section", "Unknown")
        text = chunk.get("text", "")
        context_parts.append(f"[C{i}] (Title: {title} | Section: {section})\n{text}")

    context_str = "\n\n".join(context_parts)

    user_message = f"""Context:
{context_str}

Question: {question}
Answer:"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]


def create_client(config: Dict) -> OpenAI:
    """Create OpenAI-compatible client from config."""
    gen = config.get("generation", {})
    base_url = gen.get("base_url")
    api_key = gen.get("api_key")
    model = gen.get("model")

    if not base_url or not api_key or not model:
        raise ValueError("Generation config missing base_url, api_key, or model")

    timeout = float(gen.get("timeout", 180.0))
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=3,
    )


def get_extra_body_for_ollama(base_url: str) -> Dict:
    """Return extra_body for Ollama to increase context window."""
    if "11434" in base_url:
        return {"options": {"num_ctx": 8192}}
    return {}


def generate(question: str, chunks: List[Dict], config: Dict = None, max_retries: int = None) -> Dict[str, Any]:
    """Generate an answer using the configured LLM."""
    if config is None:
        config = load_config()

    gen_cfg = config.get("generation", {})
    model = gen_cfg.get("model")
    temperature = gen_cfg.get("temperature", 0.1)
    max_tokens = int(gen_cfg.get("max_tokens", 3000))
    max_context_tokens = gen_cfg.get("max_context_tokens", 6000)
    started = time.perf_counter()
    base_url = gen_cfg.get("base_url") or ""
    extra_body = get_extra_body_for_ollama(base_url)

    # Trim chunks to fit context window
    trimmed_chunks = trim_chunks_to_context(chunks, question, max_context_tokens, config)

    # Build prompt and estimate tokens
    messages = build_prompt(question, trimmed_chunks)
    estimated_tokens = sum(estimate_tokens(m["content"]) for m in messages)

    # Debug info
    debug_info = {
        "estimated_tokens": estimated_tokens,
        "max_context_tokens": max_context_tokens,
        "max_output_tokens": max_tokens,
        "chunks_used": len(trimmed_chunks),
        "chunks_original": len(chunks),
        "prompt_messages": messages
    }

    try:
        client = create_client(config)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body if extra_body else None
        )
        msg = response.choices[0].message
        raw_content = msg.content or ""
        reasoning_content = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
        finish_reason = response.choices[0].finish_reason
        visible_answer = strip_reasoning(raw_content)
        is_abstained = "do not contain enough information" in visible_answer.lower()
        debug_info.update({
            "raw_response": raw_content,
            "reasoning_content": reasoning_content,
            "finish_reason": finish_reason,
            "usage": response.usage.model_dump() if response.usage else None,
            "is_abstained": is_abstained,
            "attempt": 1
        })
        if not visible_answer:
            raise ValueError(f"LLM returned empty response (finish_reason={finish_reason})")
        return {
            "answer": visible_answer,
            "citations": {f"C{i}": chunk["chunk_id"] for i, chunk in enumerate(trimmed_chunks, 1)},
            "raw": raw_content,
            "visible": visible_answer,
            "status": "ok",
            "debug": debug_info
        }
    except Exception as e:
        elapsed = time.perf_counter() - started
        error_msg = f"{type(e).__name__}: {e}"
        logger.error("LLM request failed after %.2fs: %s", elapsed, error_msg)
        debug_info.update({
            "error": error_msg,
            "exception_type": type(e).__name__,
            "elapsed_seconds": elapsed,
            "is_abstained": False,
        })

        return {
            "answer": "",
            "citations": {},
            "raw": "",
            "visible": "",
            "status": "llm_error",
            "error": error_msg,
            "debug": debug_info,
        }


def trim_chunks_to_context(chunks: List[Dict], question: str, max_context_tokens: int, config: Dict) -> List[Dict]:
    """Trim lowest-ranked chunks to fit within max_context_tokens."""
    if not chunks:
        return chunks

    # Estimate tokens for system prompt + question + overhead
    system_tokens = estimate_tokens(SYSTEM_PROMPT)
    question_tokens = estimate_tokens(question)
    overhead_tokens = 200  # formatting, citations, etc.

    available_tokens = max_context_tokens - system_tokens - question_tokens - overhead_tokens

    if available_tokens <= 0:
        return chunks[:1]  # At least 1 chunk

    # Try adding chunks from highest ranked (they're already ranked)
    trimmed = []
    used_tokens = 0

    for chunk in chunks:
        chunk_text = chunk.get("text", "")
        chunk_tokens = estimate_tokens(chunk_text)
        if used_tokens + chunk_tokens <= available_tokens:
            trimmed.append(chunk)
            used_tokens += chunk_tokens
        else:
            # Try to add a truncated version
            remaining = available_tokens - used_tokens
            if remaining > 100:  # Only if meaningful space
                # Truncate chunk text
                truncated_text = chunk_text[:remaining * 4]  # ~4 chars per token
                trimmed_chunk = chunk.copy()
                trimmed_chunk["text"] = truncated_text
                trimmed.append(trimmed_chunk)
            break

    return trimmed if trimmed else chunks[:1]


def parse_citations(answer: str) -> List[Tuple[str, List[str]]]:
    """
    Split answer into sentences and extract [C\\d+] tags per sentence.
    Returns list of (sentence, [citation_tags]).
    """
    if not answer or not answer.strip():
        return []

    # Handle the "not enough information" case
    if "do not contain enough information" in answer.lower():
        return [(answer.strip(), [])]

    results = []
    records = []
    citation_pattern = re.compile(r"\[C(\d+)\]")
    leading_pattern = re.compile(r"^\s*((?:\[C\d+\]\s*)+)")

    for raw_sentence in nltk.sent_tokenize(answer):
        leading_match = leading_pattern.match(raw_sentence)
        leading = citation_pattern.findall(
            leading_match.group(1)
        ) if leading_match else []
        body = raw_sentence[leading_match.end():] if leading_match else raw_sentence
        citations = leading + citation_pattern.findall(body)
        clean_sent = re.sub(r"\s*\[C\d+\]", "", body).strip()
        word_count = len(re.findall(r"[A-Za-z]+", clean_sent))
        if leading and records:
            records[-1][1].extend(f"C{citation}" for citation in leading)
        if word_count == 0:
            if not clean_sent and citations and records:
                records[-1][1].extend(
                    f"C{citation}" for citation in citations if citation not in leading
                )
            continue
        records.append([clean_sent, [f"C{citation}" for citation in citations[len(leading):]]])

    return [(sentence, citations) for sentence, citations in records]


def format_answer_with_citations(answer: str, citations_map: Dict[str, str]) -> str:
    """Replace [C1] tags with [C1: chunk_id] for display."""
    def replace_citation(match):
        tag = match.group(0)  # e.g., [C1]
        num = match.group(1)
        chunk_id = citations_map.get(tag, tag)
        return f"[{tag}: {chunk_id}]"

    return re.sub(r'\[C(\d+)\]', replace_citation, answer)


if __name__ == "__main__":
    # Demo with dummy chunks
    dummy_chunks = [
        {
            "chunk_id": "2401.01234_fixed_0001",
            "title": "Retrieval-Augmented Generation for NLP",
            "section": "Introduction",
            "text": "Retrieval-augmented generation (RAG) combines information retrieval with text generation to improve factual accuracy. It retrieves relevant documents and uses them as context for generation.",
            "n_tokens": 45,
            "strategy": "fixed",
            "paper_id": "2401.01234"
        },
        {
            "chunk_id": "2401.01234_fixed_0002",
            "title": "Retrieval-Augmented Generation for NLP",
            "section": "Method",
            "text": "The RAG pipeline consists of three components: a retriever, a reranker, and a generator. The retriever finds relevant passages using dense embeddings.",
            "n_tokens": 42,
            "strategy": "fixed",
            "paper_id": "2401.01234"
        },
        {
            "chunk_id": "2401.01234_fixed_0003",
            "title": "Retrieval-Augmented Generation for NLP",
            "section": "Experiments",
            "text": "Experiments on Natural Questions show RAG improves exact match accuracy by 15% over closed-book baselines. The improvement is larger for questions requiring multi-hop reasoning.",
            "n_tokens": 48,
            "strategy": "fixed",
            "paper_id": "2401.01234"
        }
    ]

    question = "What are the components of a RAG pipeline?"

    print("=== Generation Demo ===")
    print(f"Question: {question}\n")

    # Show prompt
    messages = build_prompt(question, dummy_chunks)
    print("System prompt:")
    print(messages[0]["content"][:200] + "...")
    print("\nUser message:")
    print(messages[1]["content"][:500] + "...")

    print("\n--- Parsing demo ---")
    # Demo parsing with a sample answer
    sample_answer = "RAG has three components: retriever, reranker, and generator [C1][C2]. The retriever uses dense embeddings [C2]."
    parsed = parse_citations(sample_answer)
    print(f"Sample answer: {sample_answer}")
    print("Parsed sentences with citations:")
    for sent, cites in parsed:
        print(f"  '{sent}' -> {cites}")