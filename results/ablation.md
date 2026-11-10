# Ablation Study

Each table averages over the other two factors.

## 1. Chunking Strategy (fixed vs semantic)

| chunking   |   recall@1 |   recall@3 |   recall@5 |   recall@10 |    mrr |   faithfulness |   hallucination |   answer_relevance |   answer_correctness |   latency |
|:-----------|-----------:|-----------:|-----------:|------------:|-------:|---------------:|----------------:|-------------------:|---------------------:|----------:|
| fixed      |     0.5726 |     0.6532 |     0.6694 |      0.7177 | 0.6207 |            nan |             nan |                nan |                  nan |       nan |
| semantic   |     0.5484 |     0.6371 |     0.6532 |      0.7016 | 0.6013 |            nan |             nan |                nan |                  nan |       nan |

## 2. Retrieval Method (dense vs hybrid)

| retrieval   |   recall@1 |   recall@3 |   recall@5 |   recall@10 |    mrr |   faithfulness |   hallucination |   answer_relevance |   answer_correctness |   latency |
|:------------|-----------:|-----------:|-----------:|------------:|-------:|---------------:|----------------:|-------------------:|---------------------:|----------:|
| dense       |     0.5000 |     0.5806 |     0.5887 |      0.6210 | 0.5444 |            nan |             nan |                nan |                  nan |       nan |
| hybrid      |     0.6210 |     0.7097 |     0.7339 |      0.7984 | 0.6776 |            nan |             nan |                nan |                  nan |       nan |

## 3. Reranking (off vs on)

| rerank   |   recall@1 |   recall@3 |   recall@5 |   recall@10 |    mrr |   faithfulness |   hallucination |   answer_relevance |   answer_correctness |   latency |
|:---------|-----------:|-----------:|-----------:|------------:|-------:|---------------:|----------------:|-------------------:|---------------------:|----------:|
| rerank0  |     0.4839 |     0.5726 |     0.6048 |      0.6855 | 0.5421 |            nan |             nan |                nan |                  nan |       nan |
| rerank1  |     0.6371 |     0.7177 |     0.7177 |      0.7339 | 0.6799 |            nan |             nan |                nan |                  nan |       nan |

