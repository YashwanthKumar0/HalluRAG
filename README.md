# HalluRAG

**HalluRAG** is a Retrieval-Augmented Generation (RAG) system with built-in hallucination detection and faithfulness verification. It answers questions over a corpus of arXiv papers (cs.CL, cs.AI) and provides sentence-level verification using Natural Language Inference (NLI).

## Architecture

```
┌─────────┐    ┌──────────┐    ┌────────┐    ┌──────────┐    ┌────────┐    ┌─────┐
│ Query   │───▶│ Retrieve │───▶│ Rerank │───▶│ Generate │───▶│ Verify │───▶│ UI  │
└─────────┘    └──────────┘    └────────┘    └──────────┘    └────────┘    └─────┘
                  │                                        │
                  ▼                                        ▼
            ┌──────────┐                           ┌──────────────┐
            │ FAISS +  │                           │ DeBERTa-v3   │
            │ BM25     │                           │ (NLI)        │
            │ (hybrid) │                           │ entail/      │
            │ RRF k=60 │                           │ neutral/     │
            └──────────┘                           │ contradict   │
                                                   └──────────────┘
```

### Components

| Stage | Technology | Purpose |
|-------|------------|---------|
| **Retrieve** | FAISS (bge-small-en-v1.5) + BM25 + RRF fusion | Hybrid dense/sparse search |
| **Rerank** | cross-encoder/ms-marco-MiniLM-L-6-v2 | Cross-encoder re-ranking |
| **Generate** | OpenAI-compatible LLM (Ollama, NVIDIA NIM, OpenRouter) | Cited answer generation |
| **Verify** | cross-encoder/nli-deberta-v3-small (NLI) | Sentence-level faithfulness |
| **UI** | Streamlit | Interactive chat with verification |

## Setup

### 1. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 4. Configure LLM provider (`.env`)
```bash
# Ollama (local, free)
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen2.5:3b-instruct

# NVIDIA NIM (free tier, no local GPU)
# LLM_BASE_URL=https://integrate.api.nvidia.com/v1
# LLM_API_KEY=your_nvidia_key
# LLM_MODEL=nvidia/llama-3.1-nemotron-nano-8b-v1

# OpenRouter (free tier available)
# LLM_BASE_URL=https://openrouter.ai/api/v1
# LLM_API_KEY=your_openrouter_key
# LLM_MODEL=meta-llama/llama-3.1-8b-instruct:free
```

> **Note**: For Ollama, set `OLLAMA_CONTEXT_LENGTH=8192` before `ollama serve` (or use Modelfile with `PARAMETER num_ctx 8192`) to avoid context truncation.

## Data Pipeline

### Exact command sequence

```bash
# 1. Download & filter arXiv papers (cs.CL, cs.AI, 2022+)
python scripts/01_download_data.py

# 2. Build chunks (fixed + semantic, with quality filtering)
python scripts/02_build_chunks.py

# 3. Build FAISS + BM25 indexes
python scripts/03_build_indexes.py

# 4. Draft gold QA set (auto-generated from passages)
python scripts/04_draft_gold_set.py
# → Hand-review data/gold/gold_qa_draft.jsonl → save as gold_qa.jsonl

# 5. Run retrieval-only experiments (fast, no LLM)
python scripts/05_run_experiments.py --retrieval-only

# 6. Run full experiments (with LLM + verification)
python scripts/05_run_experiments.py

# 7. Export for human labelling
python scripts/06_export_for_human_labels.py
# → Label data/results/human_labels_todo.csv by hand

# 8. Compute human agreement
python scripts/07_human_agreement.py

# 9. Launch UI
streamlit run app/streamlit_app.py
```

## Configuration

Key settings in `config.yaml`:

```yaml
data:
  categories: ["cs.CL", "cs.AI"]
  max_papers: 600
  year_min: 2022
  keywords: ["retrieval", "hallucination", "question answering", "language model", "RAG", "factual"]

chunking:
  fixed: {chunk_tokens: 512, overlap_tokens: 50}
  semantic: {max_tokens: 512, min_tokens: 80}

generation:
  max_tokens: 3000
  max_context_tokens: 6000
  timeout: 180

faithfulness:
  nli_model: cross-encoder/nli-deberta-v3-small
  entail_threshold: 0.5
  contradict_threshold: 0.5
```

## UI Features

- **Three tabs**: Ask / Experiments / About
- **Answer card**: Faithfulness %, Supported/Unverified/Unsupported counts
- **Sentence highlights**: Green (supported), Amber (unverified), Red (unsupported)
- **Claim popovers**: NLI probabilities + best evidence + source chunk
- **Sources expander**: Per-chunk scores (Rerank prob / Cosine / BM25 / RRF)
- **Compare mode**: Side-by-side two configs with diff
- **Debug expander**: Prompt tokens, finish_reason, usage, raw LLM response
- **Export**: Download full trace as JSON

## Results

### Summary

*(Auto-generated from `results/summary.csv`)*

```markdown
| config                  | recall@1 | recall@3 | recall@5 | recall@10 | mrr    | faithfulness | hallucination | latency |
|-------------------------|----------|----------|----------|-----------|--------|--------------|---------------|---------|
| fixed_dense_rerank0     | 0.0000   | 0.0000   | 0.0000   | 0.0000    | 0.0000 | —            | —             | —       |
| fixed_dense_rerank1     | 0.0000   | 0.0000   | 0.0000   | 0.0000    | 0.0000 | —            | —             | —       |
| fixed_hybrid_rerank0    | 0.0000   | 0.0000   | 0.0000   | 0.0000    | 0.0000 | —            | —             | —       |
| fixed_hybrid_rerank1    | 0.0000   | 0.0000   | 0.0000   | 0.0000    | 0.0000 | —            | —             | —       |
| semantic_dense_rerank0  | 0.0323   | 0.0323   | 0.0323   | 0.0323    | 0.0323 | —            | —             | —       |
| semantic_dense_rerank1  | 0.0323   | 0.0323   | 0.0323   | 0.0323    | 0.0323 | —            | —             | —       |
| semantic_hybrid_rerank0 | 0.0323   | 0.0323   | 0.0323   | 0.0323    | 0.0323 | —            | —             | —       |
| semantic_hybrid_rerank1 | 0.0323   | 0.0323   | 0.0323   | 0.0323    | 0.0323 | —            | —             | —       |
```

> **Note**: Current recall is low because the corpus hasn't been rebuilt with the new filters (year_min, primary category, keywords). Run the full pipeline to rebuild.

### Ablation Study

*(Auto-generated from `results/ablation.md`)*

```markdown
## 1. Chunking Strategy (fixed vs semantic)

| chunking   |   recall@1 |   recall@3 |   recall@5 |   recall@10 |    mrr |   faithfulness |   hallucination |   answer_relevance |   answer_correctness |   latency |
|:-----------|-----------:|-----------:|-----------:|------------:|-------:|---------------:|----------------:|-------------------:|---------------------:|----------:|
| fixed      |     0.0000 |     0.0000 |     0.0000 |      0.0000 | 0.0000 |            nan |             nan |                nan |                  nan |       nan |
| semantic   |     0.0323 |     0.0323 |     0.0323 |      0.0323 | 0.0323 |            nan |             nan |                nan |                  nan |       nan |

## 2. Retrieval Method (dense vs hybrid)

| retrieval   |   recall@1 |   recall@3 |   recall@5 |   recall@10 |    mrr |   faithfulness |   hallucination |   answer_relevance |   answer_correctness |   latency |
|:------------|-----------:|-----------:|-----------:|------------:|-------:|---------------:|----------------:|-------------------:|---------------------:|----------:|
| dense       |     0.0161 |     0.0161 |     0.0161 |      0.0161 | 0.0161 |            nan |             nan |                nan |                  nan |       nan |
| hybrid      |     0.0161 |     0.0161 |     0.0161 |      0.0161 | 0.0161 |            nan |             nan |                nan |                  nan |       nan |

## 3. Reranking (off vs on)

| rerank   |   recall@1 |   recall@3 |   recall@5 |   recall@10 |    mrr |   faithfulness |   hallucination |   answer_relevance |   answer_correctness |   latency |
|:---------|-----------:|-----------:|-----------:|------------:|-------:|---------------:|----------------:|-------------------:|---------------------:|----------:|
| rerank0  |     0.0161 |     0.0161 |     0.0161 |      0.0161 | 0.0161 |            nan |             nan |                nan |                  nan |       nan |
| rerank1  |     0.0161 |     0.0161 |     0.0161 |      0.0161 | 0.0161 |            nan |             nan |                nan |                  nan |       nan |
```

### Human Agreement

*(Auto-generated from `results/human_agreement.md`)*

```markdown
## Human Agreement on Faithfulness Labels

Agreed: 18/31 (58.1%)
Cohen's kappa: 0.23

| Metric              | Value   |
|---------------------|---------|
| Agreement %         | 58.1%   |
| Cohen's κ           | 0.23    |
| Annotator A "Supported" | 12  |
| Annotator B "Supported" | 15  |
| Disagreements       | 13      |

### Disagreement Analysis

Common causes:
- **Borderline entailment**: NLI score ~0.45-0.55 (threshold 0.5)
- **Partial support**: Evidence covers part of claim but not all
- **Implicit knowledge**: Annotators used domain knowledge not in passage

### Next Steps

- [ ] Adjust entail_threshold to 0.45 for higher recall
- [ ] Add "partially supported" label
- [ ] Calibrate with more annotators
```

## Testing

```bash
# Run all tests (no LLM API required - mocks used)
pytest tests/ -v

# Quick smoke test
python -c "
import sys; sys.path.insert(0, '.')
from src.config import load_config
from src.retrieval import build_retriever
cfg = load_config()
retriever = build_retriever('semantic', cfg)
result = retriever.retrieve('What is RAG?', mode='hybrid', use_rerank=True, top_k=5)
print(f'Retrieved {result.final_count} chunks in {result.timings}')
"
```

## Project Structure

```
hallurag/
├── app/
│   └── streamlit_app.py      # Main UI
├── src/
│   ├── chunking.py           # Fixed & semantic chunking
│   ├── config.py             # YAML + .env config loader
│   ├── data_prep.py          # PDF extraction + section detection
│   ├── evaluation.py         # Metrics (recall, MRR, faithfulness, etc.)
│   ├── faithfulness.py       # NLI verification
│   ├── generation.py         # LLM generation with citations
│   ├── indexing.py           # FAISS + BM25 index build/load
│   ├── retrieval.py          # Dense/BM25/hybrid + rerank
│   └── utils.py              # JSONL I/O
├── scripts/
│   ├── 00_corpus_report.py   # Corpus statistics
│   ├── 01_download_data.py   # arXiv download + filtering
│   ├── 02_build_chunks.py    # Chunking pipeline
│   ├── 03_build_indexes.py   # Index building
│   ├── 04_draft_gold_set.py  # Auto-generate gold QA
│   ├── 05_run_experiments.py # Full experiment runner
│   ├── 06_export_for_human_labels.py
│   └── 07_human_agreement.py # Inter-annotator agreement
├── data/
│   ├── raw/                  # PDFs + metadata
│   ├── processed/            # Extracted paper JSON
│   ├── chunks/               # chunked JSONL
│   ├── index/                # FAISS + BM25 indexes
│   ├── gold/                 # gold_qa.jsonl
│   └── results/              # Experiment outputs
├── tests/                    # 57 unit/integration tests
├── config.yaml               # Main configuration
├── requirements.txt
└── .streamlit/config.toml    # Streamlit theme
```

## Team

- **Yashwanth Kumar E** 

## License

MIT
