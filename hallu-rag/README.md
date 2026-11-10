# Hallucination-Aware RAG for Domain-Specific QA

A Retrieval-Augmented Generation (RAG) system with faithfulness verification for answering questions over arXiv papers (cs.CL / cs.AI).

## Features

- **Retrieval**: Dense (FAISS) and hybrid (BM25 + dense) retrieval with optional cross-encoder reranking
- **Generation**: Local LLM (Phi-3-mini) with inline citations `[C1]`, `[C2]`, etc.
- **Faithfulness Verification**: NLI-based (DeBERTa-v3-MNLI) sentence-level claim verification
  - 🟢 **Supported** — entailed by evidence
  - 🟡 **Unverified** — evidence is neutral
  - 🔴 **Unsupported** — evidence contradicts it
- **Evaluation**: Recall@k, MRR, faithfulness, answer relevance, human agreement (Cohen's κ)
- **Experiments**: 8 configurations (2 chunking × 2 retrieval × 2 reranking)
- **UI**: Streamlit chat interface with real-time verification display

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Copy `.env.example` to `.env` and add your HF token if needed:
```bash
cp .env.example .env
```

### 3. Prepare Data (run in order)

```bash
# Download and process arXiv papers
python scripts/01_download_data.py

# Build chunks (fixed and semantic)
python scripts/02_build_chunks.py

# Build FAISS and BM25 indexes
python scripts/03_build_indexes.py

# Create gold Q&A template (then edit it!)
python scripts/04_draft_gold_set.py
# Edit data/gold/gold_qa.jsonl with real questions

# Run experiments
python scripts/05_run_experiments.py

# Export predictions for human labeling
python scripts/06_export_for_human_labels.py
# Fill in data/gold/human_labels.jsonl

# Compute human agreement
python scripts/07_human_agreement.py
```

### 4. Launch Streamlit App

```bash
streamlit run app/streamlit_app.py
```

## Project Structure

```
hallu-rag/
├── config.yaml              # Main configuration
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
├── data/
│   ├── raw/                # Downloaded PDFs
│   ├── processed/          # Cleaned paper JSONs
│   ├── chunks/             # Chunk JSONL files
│   ├── index/              # FAISS + BM25 indexes
│   └── gold/               # Gold Q&A + human labels
├── results/                # Experiment outputs
├── src/                    # Core modules
│   ├── config.py           # Configuration loader
│   ├── data_prep.py        # PDF download & text extraction
│   ├── chunking.py         # Fixed & semantic chunking
│   ├── indexing.py         # FAISS & BM25 indexes
│   ├── retrieval.py        # Dense/hybrid/reranked retrieval
│   ├── generation.py       # Answer generation with citations
│   ├── faithfulness.py     # NLI-based verification
│   ├── pipeline.py         # End-to-end RAG pipeline
│   ├── evaluation.py       # Metrics computation
│   └── utils.py            # Utilities
├── scripts/                # Data preparation & experiment scripts
├── app/                    # Streamlit app
└── tests/                  # Unit tests
```

## Configuration

Edit `config.yaml` to customize:

- **Data**: arXiv categories, date range, paper count
- **Chunking**: Fixed (size/overlap) vs semantic (similarity threshold)
- **Retrieval**: Dense/hybrid weights, top-k, reranking model
- **Generation**: Model, temperature, max tokens, system prompt
- **Faithfulness**: NLI model, labels, confidence threshold
- **Evaluation**: Metrics, k-values, experiment grid

## Running Tests

```bash
pytest tests/ -v
```

## Citation

If you use this system, please cite:

```bibtex
@software{hallu-rag,
  title = {Hallucination-Aware RAG for Domain-Specific QA},
  year = {2024},
}
```

## License

MIT License