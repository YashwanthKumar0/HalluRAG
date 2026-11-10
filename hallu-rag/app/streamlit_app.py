"""
Streamlit chat UI for Hallucination-Aware RAG.
Phase 7: Interactive web interface.
"""
import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import get_config, reset_config
from src.pipeline import create_pipeline
from src.indexing import load_indexes
from src.utils import setup_logging


# Page config
st.set_page_config(
    page_title="Hallucination-Aware RAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_pipeline(strategy: str, retrieval_type: str, use_rerank: bool):
    """Load and cache the RAG pipeline."""
    indexes = load_indexes()
    if not indexes:
        return None

    pipeline = create_pipeline(
        strategy=strategy,
        retrieval_type=retrieval_type,
        use_rerank=use_rerank,
        indexes=indexes,
    )
    return pipeline


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "current_config" not in st.session_state:
        st.session_state.current_config = {}


def render_sidebar():
    """Render sidebar with configuration options."""
    st.sidebar.header("⚙️ Configuration")

    cfg = get_config()

    # Strategy selection
    strategy = st.sidebar.selectbox(
        "Chunking Strategy",
        ["fixed", "semantic"],
        index=0,
        help="Fixed-size chunks vs semantic chunks",
    )

    # Retrieval type
    retrieval_type = st.sidebar.selectbox(
        "Retrieval Type",
        ["dense", "hybrid"],
        index=0,
        help="Dense vector search vs BM25+dense hybrid",
    )

    # Reranking
    use_rerank = st.sidebar.checkbox(
        "Enable Cross-Encoder Reranking",
        value=True,
        help="Re-rank results with cross-encoder for better precision",
    )

    # Top-k
    top_k = st.sidebar.slider(
        "Number of Retrieved Chunks",
        min_value=1,
        max_value=20,
        value=5,
        help="How many chunks to retrieve",
    )

    # Check if config changed
    new_config = {
        "strategy": strategy,
        "retrieval_type": retrieval_type,
        "use_rerank": use_rerank,
    }

    if new_config != st.session_state.current_config:
        st.session_state.current_config = new_config
        st.session_state.pipeline = None  # Force reload
        st.rerun()

    # Load pipeline button
    if st.sidebar.button("🔄 Reload Pipeline", type="primary"):
        st.session_state.pipeline = None
        st.rerun()

    # Show current config
    st.sidebar.divider()
    st.sidebar.caption("Current Configuration:")
    st.sidebar.code(f"""
Strategy: {strategy}
Retrieval: {retrieval_type}
Rerank: {use_rerank}
Top-K: {top_k}
""")

    return strategy, retrieval_type, use_rerank, top_k


def render_faithfulness(verification: Dict[str, Any]):
    """Render faithfulness verification results."""
    st.subheader("🔍 Faithfulness Verification")

    overall = verification.get("overall_faithfulness", "unknown")
    score = verification.get("faithfulness_score", 0.0)
    supported = verification.get("supported_count", 0)
    unverified = verification.get("unverified_count", 0)
    unsupported = verification.get("unsupported_count", 0)

    # Overall status
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        color = "green" if overall == "supported" else "orange" if overall == "partially_supported" else "red"
        st.markdown(f"**Overall:** :{color}[{overall.replace('_', ' ').title()}]")
    with col2:
        st.metric("Faithfulness Score", f"{score:.2f}")
    with col3:
        st.metric("🟢 Supported", supported)
    with col4:
        st.metric("🔴 Unsupported", unsupported)

    # Claims details
    claims = verification.get("claims", [])
    if claims:
        with st.expander("View Claim-Level Details", expanded=True):
            for i, claim in enumerate(claims):
                emoji = claim.get("emoji", "⚪")
                label = claim.get("label", "unknown")
                confidence = claim.get("confidence", 0.0)
                text = claim.get("claim", "")

                label_color = "green" if label == "supported" else "orange" if label == "unverified" else "red"
                st.markdown(f"""
                **Claim {i+1}** {emoji} :{label_color}[{label.title()}] (confidence: {confidence:.2f})
                > {text}
                """)
                st.divider()


def render_citations(citations: List[Dict[str, Any]]):
    """Render citations."""
    if not citations:
        return

    st.subheader("📚 Citations")
    for cit in citations:
        with st.expander(f"[{cit['citation_id']}] {cit.get('paper_title', 'Unknown Paper')}"):
            st.caption(f"Paper ID: {cit.get('paper_id', 'N/A')}")
            st.write(cit.get('text', 'No text available'))


def render_retrieved_chunks(chunks: List[Dict[str, Any]]):
    """Render retrieved chunks."""
    if not chunks:
        return

    with st.expander("🔎 Retrieved Chunks", expanded=False):
        for chunk in chunks:
            citation_id = chunk.get("citation_id", "C?")
            score = chunk.get("score", 0)
            text = chunk.get("text", "")[:300] + "..."

            st.markdown(f"**[{citation_id}]** (score: {score:.3f})")
            st.caption(f"Paper: {chunk.get('paper_title', 'Unknown')}")
            st.write(text)
            st.divider()


def main():
    """Main Streamlit app."""
    # Initialize
    init_session_state()

    # Header
    st.title("🔬 Hallucination-Aware RAG")
    st.caption("Retrieval-Augmented Generation with Faithfulness Verification for arXiv Papers")

    # Sidebar
    strategy, retrieval_type, use_rerank, top_k = render_sidebar()

    # Load pipeline
    if st.session_state.pipeline is None:
        with st.spinner("Loading pipeline..."):
            st.session_state.pipeline = load_pipeline(strategy, retrieval_type, use_rerank)

    if st.session_state.pipeline is None:
        st.error("Failed to load pipeline. Please run the data preparation scripts first:")
        st.code("""
python scripts/01_download_data.py
python scripts/02_build_chunks.py
python scripts/03_build_indexes.py
""")
        st.stop()

    # Chat interface
    st.divider()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show verification for assistant messages
            if message["role"] == "assistant" and "verification" in message:
                render_faithfulness(message["verification"])
                render_citations(message.get("citations", []))
                render_retrieved_chunks(message.get("retrieved_chunks", []))

    # Chat input
    if prompt := st.chat_input("Ask a question about the arXiv papers..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.pipeline.run(prompt, top_k=top_k)

                    # Display answer
                    st.markdown(result.answer)

                    # Display verification
                    render_faithfulness(result.verification)
                    render_citations(result.citations)
                    render_retrieved_chunks(result.retrieved_chunks)

                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result.answer,
                        "verification": result.verification,
                        "citations": result.citations,
                        "retrieved_chunks": result.retrieved_chunks,
                    })

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())


if __name__ == "__main__":
    main()