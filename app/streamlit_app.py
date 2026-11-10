import json
import time
import textwrap
import html
import random
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

import streamlit as st
from dataclasses import dataclass

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.retrieval import build_retriever, RetrievalResult
from src.generation import generate, parse_citations
from src.faithfulness import (
    NLIScorer, verify_answer, extract_claims, normalize_citation_tags
)
from src.evaluation import run_evaluation, is_abstained as eval_is_abstained
from src.indexing import load_index


# ─── Constants & Config ───

st.set_page_config(
    page_title="HalluRAG",
    page_icon=":material/science:",
    layout="wide",
    initial_sidebar_state="expanded"
)

BADGE_MAP = {
    "green": ("Grounded", "#DCFCE7", "#166534"),
    "amber": ("Partly verified", "#FEF3C7", "#92400E"),
    "red": ("Contains unsupported claims", "#FEE2E2", "#991B1B"),
    "abstained": ("No answer found", "#F3F4F6", "#374151"),
    "error": ("LLM error", "#FEE2E2", "#991B1B"),
    "unknown": ("Unknown", "#F3F4F6", "#374151"),
}

LABEL_COLORS = {
    "supported": {"bg": "#d4edda", "text": "#155724", "border": "#c3e6cb"},
    "unverified": {"bg": "#fff3cd", "text": "#856404", "border": "#ffeeba"},
    "unsupported": {"bg": "#f8d7da", "text": "#721c24", "border": "#f5c6cb"},
}

LABEL_ICONS = {
    "supported": "✓",
    "unverified": "?",
    "unsupported": "✗",
}

REFUSAL_PHRASES = [
    "do not contain enough information",
    "not contain enough information",
    "cannot answer",
    "unable to answer"
]


# ─── Cached Resources ───

@st.cache_resource(show_spinner="Loading configuration...")
def get_config():
    return load_config()


@st.cache_resource(show_spinner="Loading NLI model...")
def get_nli_scorer(model_name: str):
    return NLIScorer(model_name)


@st.cache_resource(show_spinner="Building retriever...")
def get_retriever(chunking: str, cfg: Dict):
    return build_retriever(chunking, cfg)


@st.cache_resource(show_spinner="Loading index and models...")
def load_all_models(strategy: str, cfg: Dict):
    """Load retrieval resources; defer NLI until an answer needs verification."""
    retriever = get_retriever(strategy, cfg)
    bundle = retriever.bundle
    return bundle, retriever


# ─── Helper Functions ───

def clean_text(text: str) -> str:
    """Clean text from PDF artifacts: ligatures, special chars, extra whitespace."""
    if not text:
        return ""
    ligatures = {
        'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬃ': 'ffi', 'ﬄ': 'ffl', 'ﬀ': 'ff',
        '‘': "'", '’': "'", '“': '"', '”': '"', '–': '-', '—': '--',
        '…': '...', '\xa0': ' ',
    }
    for k, v in ligatures.items():
        text = text.replace(k, v)
    text = ' '.join(text.split())
    return text


def truncate_text(text: str, max_len: int = 600) -> str:
    """Truncate text at word boundary."""
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    last_space = truncated.rfind(' ')
    if last_space > max_len * 0.8:
        return truncated[:last_space] + '...'
    return truncated + '...'


def render_badge(overall_flag: str, faithfulness: Optional[float], abstained: bool = False) -> None:
    """Render the overall faithfulness badge without shortcode rendering."""
    flag = "abstained" if abstained else overall_flag
    label, bg, fg = BADGE_MAP.get(flag, BADGE_MAP["unknown"])

    faith_text = f" — Faithfulness: {faithfulness:.1%}" if faithfulness is not None else ""
    st.html(
        f'<span style="background:{bg};color:{fg};padding:4px 12px;'
        f'border-radius:999px;font-size:13px;font-weight:500">'
        f'{html.escape(label + faith_text)}</span>'
    )


def render_sentence(sentence: str, label: str, probs: Dict = None, evidence: str = "") -> None:
    """Render a single sentence with colored background using st.html."""
    sentence = clean_text(sentence)
    evidence = clean_text(evidence)

    colors = LABEL_COLORS.get(label, {"bg": "#ffffff", "text": "#000000", "border": "#cccccc"})
    icon = LABEL_ICONS.get(label, "")

    hover_parts = []
    if probs:
        hover_parts.append(f"Entail: {probs.get('entail', 0):.3f}")
        hover_parts.append(f"Neutral: {probs.get('neutral', 0):.3f}")
        hover_parts.append(f"Contradict: {probs.get('contradict', 0):.3f}")
    if evidence:
        hover_parts.append(f"Evidence: {evidence[:300]}...")

    hover_text = " | ".join(hover_parts) if hover_parts else ""

    label_display = {
        "supported": "Supported",
        "unverified": "Unverified",
        "unsupported": "Unsupported"
    }.get(label, label)

    safe_sentence = html.escape(sentence)
    safe_hover = html.escape(hover_text)
    safe_label = html.escape(label_display)

    html_content = textwrap.dedent(f"""
    <div style="
        background-color: {colors['bg']};
        border-left: 4px solid {colors['border']};
        color: {colors['text']};
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 6px;
        font-family: inherit;
        line-height: 1.8;
        font-size: 0.95rem;
    " title="{safe_hover}">
        <span style="font-weight: 600; font-size: 0.8rem; opacity: 0.85; margin-right: 8px;">
            {icon} {safe_label}
        </span>
        <span>{safe_sentence}</span>
    </div>
    """).strip()

    st.html(html_content)


def render_sources(citations: Dict[str, str], retrieved_chunks: List[Dict]) -> None:
    """Render sources using native Streamlit components."""
    chunk_by_id = {c["chunk_id"]: c for c in retrieved_chunks}

    with st.expander("Sources", expanded=False):
        for cite_tag, chunk_id in citations.items():
            chunk = chunk_by_id.get(chunk_id)
            if not chunk:
                continue

            paper_id = chunk.get("paper_id", "unknown")
            title = clean_text(chunk.get("title", "Unknown"))
            section = clean_text(chunk.get("section", "Unknown"))
            text = clean_text(chunk.get("text", ""))
            score = chunk.get("score", 0.0)
            score_type = chunk.get("score_type", "unknown")
            arxiv_url = f"https://arxiv.org/abs/{paper_id}"

            # Format score label
            score_label = "Score"
            if score_type == "rerank":
                prob = 1.0 / (1.0 + np.exp(-score)) if abs(score) < 20 else (1.0 if score > 0 else 0.0)
                score_label = f"Rerank score: {score:.4f} (prob {prob:.2f})"
            elif score_type == "cosine":
                score_label = f"Cosine similarity: {score:.4f}"
            elif score_type == "bm25":
                score_label = f"BM25 score: {score:.4f}"
            elif score_type == "rrf":
                score_label = f"RRF score: {score:.4f}"

            with st.container(border=True):
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown(f"[{cite_tag}]")
                    st.caption(score_label)
                with col2:
                    st.markdown(f"**[{title}]({arxiv_url})**")
                    st.caption(f"Section: {section} | Paper: {paper_id}")

                    # Show full chunk toggle
                    show_full = st.checkbox("Show full chunk", key=f"full_{chunk_id}", value=False)
                    display_text = text if show_full else truncate_text(text, 600)
                    st.text(display_text)


def format_timings(timings: Dict) -> str:
    """Format timings for display."""
    parts = []
    for key, label in [("model_loading", "Model loading (first query only)"), ("retrieval", "Retrieval"),
                       ("dense", "Query/FAISS"), ("bm25", "BM25"), ("fusion", "Fusion"),
                       ("rerank", "Rerank"),
                       ("generation", "Generation"), ("verification", "Verification")]:
        if key in timings:
            parts.append(f"{label}: {timings[key]:.2f}s")
    return " | ".join(parts)


def get_example_questions() -> List[str]:
    """Load example questions from gold QA file."""
    gold_path = Path("data/gold/gold_qa.jsonl")
    if gold_path.exists():
        questions = []
        with open(gold_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if item.get("answer") is not None:  # Only answerable
                        questions.append(item["question"])
                except:
                    pass
        if questions:
            random.seed(0)
            return random.sample(questions, min(4, len(questions)))

    # Fallback hardcoded
    return [
        "What does the quantitative model produce when it encounters something unquantifiable?",
        "Why are exact belief updates considered impractical in POMDPs?",
        "What are the three contributions of the paper 'Predefined Sparseness in Recurrent Sequence Models'?",
        "What is the formula for calculating the influenced neighbors score (INS) of a vertex v?"
    ]


def build_config_pill(chunking: str, retrieval: str, rerank: bool, top_k: int) -> str:
    """Build config summary pill."""
    rerank_str = "rerank on" if rerank else "rerank off"
    return f"{chunking} · {retrieval} · {rerank_str} · k={top_k}"


@dataclass
class AskResult:
    """Result from HalluRAG pipeline."""
    answer: str
    citations: Dict[str, str]
    retrieved_chunks: List[Dict]
    verification: Dict[str, Any]
    timings: Dict[str, float]
    config: Dict[str, Any]
    rerank_info: str = ""
    debug: Dict = None
    error: str = None
    status: str = "ok"


def run_hallurag(question: str, chunking: str, retrieval: str, use_rerank: bool,
                 top_k: int, candidate_k: int, entail_thresh: float, contradict_thresh: float,
                 cfg: Dict) -> AskResult:
    """Run the full HalluRAG pipeline with timing."""
    timings = {}

    # Load models once; keep first-query startup separate from per-query timings.
    model_start = time.perf_counter()
    bundle, retriever = load_all_models(chunking, cfg)
    timings["model_loading"] = time.perf_counter() - model_start

    # 1. Retrieval
    start = time.perf_counter()
    result: RetrievalResult = retriever.retrieve(
        query=question,
        mode=retrieval,
        use_rerank=use_rerank,
        top_k=top_k
    )
    timings["retrieval"] = time.perf_counter() - start
    for stage in ("dense", "bm25", "fusion"):
        if stage in result.timings:
            timings[stage] = result.timings[stage]

    # 2. Reranking (from retriever timings)
    timings["rerank"] = result.timings.get("rerank", 0.0)

    # 3. Generation
    start = time.perf_counter()
    top_k_chunks = result.chunks[:top_k]
    answer_data = generate(question, top_k_chunks, cfg)
    timings["generation"] = time.perf_counter() - start

    # 4. Verification
    start = time.perf_counter()
    faithfulness_cfg = cfg.get("faithfulness", {}).copy()
    faithfulness_cfg["entail_threshold"] = entail_thresh
    faithfulness_cfg["contradict_threshold"] = contradict_thresh

    # Use the cleaned "visible" answer for abstention check
    raw_answer = answer_data.get("answer", "")
    visible_answer = answer_data.get("visible", raw_answer)
    error = answer_data.get("error")
    status = answer_data.get("status", "ok")
    is_abstention = eval_is_abstained(visible_answer)

    if status == "llm_error":
        verification = {
            "claims": [],
            "faithfulness": None,
            "n_supported": 0,
            "n_unverified": 0,
            "n_unsupported": 0,
            "abstained": False,
            "overall_flag": "error",
        }
    elif is_abstention:
        verification = {
            "claims": [],
            "faithfulness": None,
            "n_supported": 0,
            "n_unverified": 0,
            "n_unsupported": 0,
            "abstained": True,
            "overall_flag": "abstained"
        }
    else:
        nli_scorer = get_nli_scorer(
            cfg.get("faithfulness", {}).get(
                "nli_model",
                "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
            )
        )
        verification = verify_answer(
            answer=visible_answer,
            citations=answer_data.get("citations", {}),
            retrieved_chunks=result.chunks,
            nli_scorer=nli_scorer,
            cfg=faithfulness_cfg,
            relevance_scorer=retriever.cross_encoder,
        )
        if error:
            verification["overall_flag"] = "error"
    timings["verification"] = time.perf_counter() - start
    answer_data.setdefault("debug", {}).update({
        "nli_pairs_scored": verification.get("nli_pairs_scored", 0),
        "claim_timings": verification.get("claim_timings", []),
        "claim_count": len(verification.get("claims", [])),
        "normalized_citation_answer": normalize_citation_tags(
            answer_data.get("debug", {}).get("raw_response", visible_answer)
        ),
    })

    # Build rerank info
    rerank_info = ""
    if result.reranked:
        rerank_info = f"Reranked {result.candidate_count} → {result.final_count}"

    return AskResult(
        answer=visible_answer,
        citations=answer_data.get("citations", {}),
        retrieved_chunks=result.chunks,
        verification=verification,
        timings=timings,
        config={
            "chunking": chunking,
            "retrieval": retrieval,
            "rerank": use_rerank,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "entail_threshold": entail_thresh,
            "contradict_threshold": contradict_thresh,
        },
        rerank_info=rerank_info,
        debug=answer_data.get("debug", {}),
        error=error
        ,status=status
    )


def render_answer_card(result: AskResult, key_prefix: str = "") -> None:
    """Render the full answer card with metrics, sentences, claims, sources."""
    verification = result.verification
    is_abstained = verification.get("abstained", False)
    overall_flag = verification.get("overall_flag", "amber")
    faithfulness = verification.get("faithfulness")
    claims = verification.get("claims", [])

    n_supported = verification.get("n_supported", 0)
    n_unverified = verification.get("n_unverified", 0)
    n_unsupported = verification.get("n_unsupported", 0)
    total_claims = n_supported + n_unverified + n_unsupported
    faith_pct = f"{faithfulness:.1%}" if faithfulness is not None else "N/A"

    # --- Top row: 4 metric tiles + badge ---
    col_badge, col_metrics = st.columns([2, 4])  # Give badge more width
    with col_badge:
        render_badge(overall_flag, faithfulness, is_abstained)
    if result.status != "llm_error":
        with col_metrics:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Faithfulness", faith_pct)
            m2.metric("Supported", n_supported)
            m3.metric("Unverified", n_unverified)
            m4.metric("Unsupported", n_unsupported)

    st.divider()

    # --- Answer sentences ---
    if result.status == "llm_error":
        st.error(f"LLM error: {result.error}")
    elif is_abstained:
        st.info("The provided documents do not contain enough information to answer this question. Try rephrasing or asking about a different topic.")
    else:
        parsed = parse_citations(result.answer)
        for i, (sentence, cite_tags) in enumerate(parsed):
            # Find matching claim
            claim_info = None
            for claim in claims:
                if claim["claim"] in sentence or sentence in claim["claim"]:
                    claim_info = claim
                    break

            if claim_info:
                label = claim_info.get("label", "unverified")
                probs = {
                    "entail": claim_info.get("entail", 0),
                    "neutral": claim_info.get("neutral", 0),
                    "contradict": claim_info.get("contradict", 0)
                }
                evidence = claim_info.get("best_evidence", "")
            else:
                label = "unverified"
                probs = {"entail": 0, "neutral": 0, "contradict": 0}
                evidence = ""

            # Render sentence with citations appended
            cite_str = "".join(
                tag if tag.startswith("[") else f"[{tag}]"
                for tag in cite_tags
            )
            full_sentence = f"{sentence} {cite_str}" if cite_str else sentence
            render_sentence(full_sentence, label, probs, evidence)

    # --- Claim Details ---
    if claims and not is_abstained and result.status != "llm_error":
        st.subheader("Claim Details", divider="gray")
        for i, claim in enumerate(claims):
            label = claim.get("label", "unverified")
            probs = {
                "entail": claim.get("entail", 0),
                "neutral": claim.get("neutral", 0),
                "contradict": claim.get("contradict", 0)
            }
            evidence = claim.get("best_evidence", "")
            chunk_id = claim.get("best_chunk_id", "")

            with st.popover(f"Claim {i+1}: {label.capitalize()} ({LABEL_ICONS.get(label, '')})", width="stretch"):
                st.markdown(f"**Claim:** {html.escape(claim.get('claim', ''))}")
                st.markdown("**NLI Probabilities:**")
                col_e, col_n, col_c = st.columns(3)
                col_e.progress(probs['entail'], text=f"Entail: {probs['entail']:.3f}")
                col_n.progress(probs['neutral'], text=f"Neutral: {probs['neutral']:.3f}")
                col_c.progress(probs['contradict'], text=f"Contradict: {probs['contradict']:.3f}")
                if evidence:
                    st.markdown("**Best Evidence:**")
                    st.markdown(
                        f'<div style="background:#FEF3C7;border-left:4px solid #F59E0B;'
                        f'padding:8px 12px;border-radius:4px">'
                        f'{html.escape(evidence[:500])}</div>',
                        unsafe_allow_html=True,
                    )
                if chunk_id:
                    st.caption(f"From chunk: {chunk_id}")
                top_windows = claim.get("top_windows", [])
                if top_windows:
                    st.markdown("**Top relevant evidence windows:**")
                    for window in top_windows:
                        st.caption(
                            f"{window['score']:.3f} — "
                            f"{window['evidence'][:300]}"
                        )

    # --- Sources ---
    render_sources(result.citations, result.retrieved_chunks)

    # --- Copy answer expander ---
    with st.expander("Copy answer"):
        st.code(result.answer, language=None, wrap_lines=True)

    # --- Error display ---
    if result.error and result.status != "llm_error":
        st.error(f"LLM Error: {result.error}")

    # --- Debug expander ---
    if result.debug:
        with st.expander("Debug: Prompt sent to LLM", expanded=False):
            debug = result.debug
            st.write(f"**Estimated tokens:** {debug.get('estimated_tokens', 'N/A')}")
            st.write(f"**Max context tokens:** {debug.get('max_context_tokens', 'N/A')}")
            st.write(f"**Max output tokens:** {debug.get('max_output_tokens', 'N/A')}")
            st.write(f"**Chunks used:** {debug.get('chunks_used', 'N/A')} / {debug.get('chunks_original', 'N/A')}")
            if debug.get('mock'):
                st.write("**Mock answer used:** Yes (LLM unavailable)")
            if debug.get('is_abstained') is not None:
                st.write(f"**Abstained:** {debug.get('is_abstained')}")
            if debug.get('finish_reason'):
                st.write(f"**Finish reason:** {debug.get('finish_reason')}")
            if debug.get('usage'):
                st.write(f"**Usage:** {debug.get('usage')}")
            st.write(f"**Claim count:** {debug.get('claim_count', 0)}")
            raw_col, normalized_col = st.columns(2)
            with raw_col:
                st.markdown("**Raw LLM answer**")
                st.code(debug.get("raw_response", "") or "(empty)")
            with normalized_col:
                st.markdown("**After citation normalization**")
                st.code(
                    debug.get("normalized_citation_answer", "") or "(empty)"
                )
            st.write(f"**NLI pairs scored:** {debug.get('nli_pairs_scored', 0)}")
            for claim_timing in debug.get("claim_timings", []):
                st.write(
                    f"- {claim_timing['seconds']:.2f}s, "
                    f"{claim_timing['pairs']} pairs: {claim_timing['claim']}"
                )
            st.write(f"**Reasoning content (first 400 chars):**")
            st.code(debug.get('reasoning_content', '')[:400] or '(none)')
            st.write("**Prompt messages:**")
            for msg in debug.get("prompt_messages", []):
                st.write(f"- **{msg['role']}**: {msg['content'][:200]}...")

    # --- Footer: timings, download ---
    st.divider()
    col_time, col_download = st.columns([2, 1])
    with col_time:
        st.caption(f"⏱️ {format_timings(result.timings)}")
        if result.rerank_info:
            st.caption(f"🔄 {result.rerank_info}")
    with col_download:
        report = {
            "question": st.session_state.get("current_question", ""),
            "answer": result.answer,
            "citations": result.citations,
            "claims": [
                {
                    "claim": c.get("claim", ""),
                    "label": c.get("label", ""),
                    "entail": c.get("entail", 0),
                    "neutral": c.get("neutral", 0),
                    "contradict": c.get("contradict", 0),
                    "best_evidence": c.get("best_evidence", ""),
                    "best_chunk_id": c.get("best_chunk_id", "")
                } for c in claims
            ],
            "verification": {
                "faithfulness": faithfulness,
                "overall_flag": overall_flag,
                "n_supported": n_supported,
                "n_unverified": n_unverified,
                "n_unsupported": n_unsupported,
                "abstained": is_abstained
            },
            "timings": result.timings,
            "config": result.config,
            "rerank_info": result.rerank_info
            ,"status": result.status
        }
        st.download_button(
            "Download JSON",
            data=json.dumps(report, indent=2),
            file_name=f"hallurag_report_{int(time.time())}.json",
            mime="application/json",
            width="stretch"
        )


def render_comparison(result_a: AskResult, result_b: AskResult, config_a: Dict, config_b: Dict) -> None:
    """Render side-by-side comparison of two configs."""
    st.subheader("Comparison", divider="blue")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Config A:** {build_config_pill(**config_a)}")
        render_answer_card(result_a, "a")
    with col_b:
        st.markdown(f"**Config B:** {build_config_pill(**config_b)}")
        render_answer_card(result_b, "b")

    # One-line diff
    faith_a = result_a.verification.get("faithfulness", 0) or 0
    faith_b = result_b.verification.get("faithfulness", 0) or 0
    unsup_a = result_a.verification.get("n_unsupported", 0)
    unsup_b = result_b.verification.get("n_unsupported", 0)
    diff_faith = (faith_b - faith_a) * 100
    diff_unsup = unsup_b - unsup_a
    st.caption(f"Config B: {diff_faith:+.1f}% faithfulness, {diff_unsup:+d} unsupported claims")


# ─── Main App ───

def main():
    # Load config
    cfg = get_config()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "compare_mode" not in st.session_state:
        st.session_state.compare_mode = False
    if "compare_config_a" not in st.session_state:
        st.session_state.compare_config_a = {}
    if "compare_config_b" not in st.session_state:
        st.session_state.compare_config_b = {}
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""

    # ─── Sidebar ───
    with st.sidebar:
        st.title("HalluRAG")
        st.caption("Answers you can verify, claim by claim")
        st.divider()

        # Compare mode toggle
        st.session_state.compare_mode = st.toggle("Compare two configs", value=st.session_state.compare_mode)

        if st.session_state.compare_mode:
            st.subheader("Config A")
            chunking_a = st.selectbox("Chunking", ["semantic", "fixed"], index=0, key="chunking_a")
            retrieval_a = st.selectbox("Retrieval", ["hybrid", "dense", "bm25"], index=0, key="retrieval_a")
            rerank_a = st.checkbox("Rerank", value=True, key="rerank_a")
            top_k_a = st.slider("Top-K", 3, 10, 5, key="top_k_a")
            candidate_k_a = st.slider("Candidate K", 10, 50, 20, key="candidate_k_a")

            st.subheader("Config B")
            chunking_b = st.selectbox("Chunking", ["semantic", "fixed"], index=1, key="chunking_b")
            retrieval_b = st.selectbox("Retrieval", ["hybrid", "dense", "bm25"], index=0, key="retrieval_b")
            rerank_b = st.checkbox("Rerank", value=False, key="rerank_b")
            top_k_b = st.slider("Top-K", 3, 10, 5, key="top_k_b")
            candidate_k_b = st.slider("Candidate K", 10, 50, 20, key="candidate_k_b")

            # Common verification settings
            st.subheader("Verification (both)")
            entail_thresh = st.slider("Entail threshold", 0.0, 1.0,
                                     cfg.get("faithfulness", {}).get("entail_threshold", 0.5), 0.05)
            contradict_thresh = st.slider("Contradict threshold", 0.0, 1.0,
                                         cfg.get("faithfulness", {}).get("contradict_threshold", 0.5), 0.05)

            st.session_state.compare_config_a = {
                "chunking": chunking_a, "retrieval": retrieval_a, "rerank": rerank_a,
                "top_k": top_k_a, "candidate_k": candidate_k_a,
                "entail_threshold": entail_thresh, "contradict_threshold": contradict_thresh
            }
            st.session_state.compare_config_b = {
                "chunking": chunking_b, "retrieval": retrieval_b, "rerank": rerank_b,
                "top_k": top_k_b, "candidate_k": candidate_k_b,
                "entail_threshold": entail_thresh, "contradict_threshold": contradict_thresh
            }
        else:
            st.subheader("Retrieval")
            chunking = st.selectbox("Chunking Strategy", ["semantic", "fixed"], index=0,
                                   help="Semantic: paragraph/sentence boundaries. Fixed: sliding window.")
            retrieval = st.selectbox("Retrieval Method", ["hybrid", "dense", "bm25"], index=0,
                                    help="Hybrid: RRF fusion of dense + BM25. Dense: FAISS only. BM25: keyword only.")
            rerank = st.checkbox("Enable Reranking", value=True,
                                help="Cross-encoder reranking of top candidates")
            top_k = st.slider("Top-K Results", 3, 10,
                             cfg.get("retrieval", {}).get("top_k", 5),
                             help="Number of chunks to use for generation")
            candidate_k = st.slider("Candidate K", 10, 50,
                                   cfg.get("retrieval", {}).get("candidate_k", 20),
                                   help="Number of candidates before reranking")

            st.subheader("Verification")
            entail_thresh = st.slider("Entailment Threshold", 0.0, 1.0,
                                     cfg.get("faithfulness", {}).get("entail_threshold", 0.5), 0.05)
            contradict_thresh = st.slider("Contradiction Threshold", 0.0, 1.0,
                                         cfg.get("faithfulness", {}).get("contradict_threshold", 0.5), 0.05)

        st.divider()

        # Corpus stats
        st.subheader("Corpus Stats")
        try:
            chunks_fixed = Path("data/chunks/chunks_fixed.jsonl")
            chunks_semantic = Path("data/chunks/chunks_semantic.jsonl")
            if chunks_fixed.exists():
                n_fixed = sum(1 for _ in open(chunks_fixed))
                st.caption(f"Fixed chunks: {n_fixed:,}")
            if chunks_semantic.exists():
                n_semantic = sum(1 for _ in open(chunks_semantic))
                st.caption(f"Semantic chunks: {n_semantic:,}")

            # Year range from processed papers
            processed_dir = Path(cfg["data"]["processed_dir"])
            if processed_dir.exists():
                papers = list(processed_dir.glob("*.json"))
                years = []
                for p in papers:
                    pid = p.stem
                    if re.match(r'^\d{4}\.', pid):
                        year = int(pid[:2])
                        years.append(2000 + year if year < 50 else 1900 + year)
                if years:
                    st.caption(f"Year range: {min(years)}–{max(years)}")
        except:
            pass

        st.divider()

        if st.button("Clear Chat", width="stretch"):
            st.session_state.messages = []
            st.rerun()

    # ─── Main Content: Three Tabs ───
    tab_ask, tab_experiments, tab_about = st.tabs([
        "Ask",
        "Experiments",
        "About"
    ])

    # ─── TAB: Ask ───
    with tab_ask:
        # Header row
        col_title, col_config = st.columns([3, 2])
        with col_title:
            st.title("HalluRAG")
            st.caption("Answers you can verify, claim by claim")
        with col_title:
            if not st.session_state.compare_mode:
                config_pill = build_config_pill(chunking, retrieval, rerank, top_k)
                st.markdown(f"**Config:** `{config_pill}`")

        # Empty state with example questions
        if not st.session_state.messages:
            st.markdown("### Try an example question")
            examples = get_example_questions()
            cols = st.columns(2)
            for i, q in enumerate(examples):
                with cols[i % 2]:
                    label = q[:70] + "…" if len(q) > 70 else q
                    if st.button(label, key=f"ex_{i}", width="stretch", help=q):
                        st.session_state.pending_question = q
                        st.rerun()

        # Chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(msg["content"])
                else:
                    if st.session_state.compare_mode and "result_b" in msg:
                        render_comparison(msg["result_a"], msg["result_b"],
                                         msg["config_a"], msg["config_b"])
                    else:
                        render_answer_card(msg["result"])

        # Chat input - check pending_question first
        prompt = st.chat_input("Ask a question about the papers...")

        # Also check for pending question from example buttons
        if not prompt:
            prompt = st.session_state.pop("pending_question", None)

        if prompt:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            # Process with HalluRAG
            with st.chat_message("assistant"):
                # Use st.status for progress
                with st.status("Answering...", expanded=True) as status:
                    try:
                        if st.session_state.compare_mode:
                            # Run both configs
                            config_a = st.session_state.compare_config_a
                            config_b = st.session_state.compare_config_b

                            status.update(label="Running Config A...")
                            result_a = run_hallurag(
                                prompt,
                                chunking=config_a["chunking"],
                                retrieval=config_a["retrieval"],
                                use_rerank=config_a["rerank"],
                                top_k=config_a["top_k"],
                                candidate_k=config_a["candidate_k"],
                                entail_thresh=config_a["entail_threshold"],
                                contradict_thresh=config_a["contradict_threshold"],
                                cfg=cfg
                            )

                            status.update(label="Running Config B...")
                            result_b = run_hallurag(
                                prompt,
                                chunking=config_b["chunking"],
                                retrieval=config_b["retrieval"],
                                use_rerank=config_b["rerank"],
                                top_k=config_b["top_k"],
                                candidate_k=config_b["candidate_k"],
                                entail_thresh=config_b["entail_threshold"],
                                contradict_thresh=config_b["contradict_threshold"],
                                cfg=cfg
                            )

                            status.update(label="Complete!", state="complete", expanded=False)

                            # Store both results
                            msg_data = {
                                "role": "assistant",
                                "result_a": result_a,
                                "result_b": result_b,
                                "config_a": config_a,
                                "config_b": config_b
                            }
                            render_comparison(result_a, result_b, config_a, config_b)

                        else:
                            # Single config
                            status.update(label="Retrieving...")
                            result = run_hallurag(
                                prompt,
                                chunking=chunking,
                                retrieval=retrieval,
                                use_rerank=rerank,
                                top_k=top_k,
                                candidate_k=candidate_k,
                                entail_thresh=entail_thresh,
                                contradict_thresh=contradict_thresh,
                                cfg=cfg
                            )

                            status.update(label="Complete!", state="complete", expanded=False)

                            # Store single result
                            msg_data = {
                                "role": "assistant",
                                "result": result
                            }
                            render_answer_card(result)

                        st.session_state.messages.append(msg_data)

                    except Exception as e:
                        status.update(label="Error", state="error", expanded=True)
                        st.exception(e)
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

    # ─── TAB: Experiments ───
    with tab_experiments:
        st.header("Experiment Results")
        results_dir = Path("results")

        if not results_dir.exists():
            st.info("No results directory found. Run `python scripts/05_run_experiments.py` to generate results.")
        else:
            # Summary table
            summary_path = results_dir / "summary.csv"
            if summary_path.exists():
                import pandas as pd
                df = pd.read_csv(summary_path)

                st.subheader("Summary Table")
                # Format columns
                format_dict = {}
                for col in df.columns:
                    if col not in ["config", "n_questions", "n_answered"]:
                        if "recall" in col.lower() or "mrr" in col.lower() or "faithfulness" in col.lower() or "hallucination" in col.lower() or "relevance" in col.lower() or "correctness" in col.lower():
                            format_dict[col] = "{:.4f}"
                        elif "latency" in col.lower():
                            format_dict[col] = "{:.2f}s"

                st.dataframe(df.style.format(format_dict).highlight_max(axis=0, subset=[c for c in df.columns if c != "config"]), width="stretch")

                # Bar charts
                st.subheader("Retrieval Metrics")
                chart_cols = st.columns(2)
                with chart_cols[0]:
                    st.bar_chart(df.set_index("config")[["recall@5", "recall@10"]])
                with chart_cols[1]:
                    st.bar_chart(df.set_index("config")["mrr"])

                st.subheader("Faithfulness Metrics")
                faith_cols = [c for c in df.columns if "faith" in c.lower() or "hallucin" in c.lower()]
                if faith_cols:
                    st.bar_chart(df.set_index("config")[faith_cols])
            else:
                st.info("No summary.csv found in results/")

            # Ablation tables
            ablation_path = results_dir / "ablation.md"
            if ablation_path.exists():
                with st.expander("Ablation Study", expanded=False):
                    st.markdown(ablation_path.read_text())

            # Failure cases
            failure_path = results_dir / "failure_cases.md"
            if failure_path.exists():
                with st.expander("Failure Cases", expanded=False):
                    st.markdown(failure_path.read_text())

            # Human agreement
            agreement_path = results_dir / "human_agreement.md"
            if agreement_path.exists():
                with st.expander("Human Agreement", expanded=True):
                    content = agreement_path.read_text()
                    st.markdown(content)
                    # Try to extract metrics for display
                    agree_match = re.search(r'Agreed:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)', content)
                    kappa_match = re.search(r"Cohen's kappa:\s*([\d.]+)", content)
                    if agree_match:
                        agreed, total, pct = agree_match.groups()
                        st.metric("Agreement", f"{pct}%", f"{agreed}/{total}")
                    if kappa_match:
                        kappa = float(kappa_match.group(1))
                        st.metric("Cohen's κ", f"{kappa:.3f}", delta="PASS" if kappa >= 0.6 else "FAIL")

    # ─── TAB: About ───
    with tab_about:
        st.header("About HalluRAG")

        st.markdown("""
        **HalluRAG** is a Retrieval-Augmented Generation system with built-in hallucination detection
        and faithfulness verification. It answers questions over a corpus of arXiv papers (cs.CL, cs.AI)
        and provides sentence-level verification using Natural Language Inference (NLI).
        """)

        st.subheader("Pipeline")
        # Graphviz diagram
        st.graphviz_chart("""
        digraph G {
            rankdir=LR;
            node [fontname="sans-serif", fontsize=11, shape=box, style="rounded,filled", fillcolor="#f0f4f8", color="#4F46E5", fontcolor="#1e293b"];
            edge [color="#64748b", fontname="sans-serif", fontsize=10];

            Query [fillcolor="#e0e7ff"];
            Retrieve [fillcolor="#dbeafe"];
            Rerank [fillcolor="#dbeafe"];
            Generate [fillcolor="#fef3c7"];
            Verify [fillcolor="#dcfce7"];
            UI [fillcolor="#fce7f3"];

            Query -> Retrieve -> Rerank -> Generate -> Verify -> UI;
        }
        """)

        st.subheader("Verification Legend")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div style="background:#d4edda;border-left:4px solid #c3e6cb;padding:10px;border-radius:6px;">
            <strong>✓ Supported</strong><br>
            Claim entailed by sources
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style="background:#fff3cd;border-left:4px solid #ffeeba;padding:10px;border-radius:6px;">
            <strong>? Unverified</strong><br>
            Not enough evidence either way
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div style="background:#f8d7da;border-left:4px solid #f5c6cb;padding:10px;border-radius:6px;">
            <strong>✗ Unsupported</strong><br>
            Claim contradicted by sources
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div style="background:#e2e3e5;border-left:4px solid #d6d8db;padding:10px;border-radius:6px;">
            <strong>ℹ No answer</strong><br>
            Corpus lacks relevant information
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.subheader("Team")
        st.markdown("""
        - **Yashwanth Kumar E** (ENG24AM0115)
        - **T Vishnu Karthik** (ENG24AM0301)
        """)

if __name__ == "__main__":
    main()