# Objectives 1-6 Implementation and Data Evidence

This document summarizes how the six project objectives are implemented in the chatbot project, what data supports each objective, and which charts or metrics can be produced from the current repository.

## Current Evidence Snapshot

The project now includes a repeatable offline evaluation workflow and dashboard evidence beyond the live chatbot demo.

| Area | Current Evidence |
| --- | --- |
| Knowledge base coverage | YC, HK Companies Registry, business guides, and social media sentiment assets |
| Topic routing | Offline test set with classification report and confusion matrix |
| Retrieval ranking | Recall@K, MRR, NDCG@5, and Top-K retrieval samples |
| Answer summarization | Generated-answer evaluation with ROUGE-style metrics |
| Follow-up prediction | Clickable follow-up suggestions, shown/clicked event logging, CTR summary |
| Dashboard analytics | Streamlit dashboard with data coverage, evaluation metrics, sentiment analytics, and follow-up analytics |

Latest generated-answer evaluation:

```text
Test queries: 22
Answer metric mode: generated_answer
Generated answers: 22/22
Topic Accuracy: 100.0%
Weighted F1: 100.0%
Relevant Recall@5: 94.44%
Relevant MRR: 0.8889
Relevant NDCG@5: 0.9034
ROUGE-1: 0.1446
ROUGE-2: 0.0406
ROUGE-L: 0.1104
Average retrieval latency: 605.32 ms
```

## Objective 1: Knowledge Base and Keyword Extraction

### Goal

Show the breadth and depth of data collected from YC, HK Companies Registry, business guides, and social media.

### Implementation

The project builds and stores structured knowledge assets in `data/kb/` and `data/sentiment/`.

Main files:

```text
data/kb/summary.json
data/kb/cleaned_company_kb.jsonl
data/kb/knowledge_base_chunks.jsonl
data/kb/hk_official_kb_chunks.jsonl
data/kb/business_guide_kb_chunks.jsonl
data/kb/global_keywords.csv
data/kb/company_keywords.csv
data/kb/topic_taxonomy.json
data/sentiment/sentiment_summary.json
data/sentiment/sentiment_cleaned.jsonl
data/sentiment/sentiment_keyword_profiles.json
```

### Data Evidence

| Data Source | Current Size |
| --- | ---: |
| YC knowledge base | 5,494 company/chunk rows |
| HK Companies Registry official guidance | 54 chunks |
| Business guide pages | 33 chunks |
| Social media sentiment dataset | 732 rows |
| YC topic clusters | 20 topics |

### Charts and Tables Available

| Requested Evidence | Status | Data Source |
| --- | --- | --- |
| Word cloud / TF-IDF Top 20 bar chart | Available | `global_keywords.csv`, `company_keywords.csv` |
| Sunburst by source and sub-category | Available | KB chunk metadata |
| Data source summary table | Available | KB JSONL files and summaries |
| Keyword density / vocabulary size | Available through scripts or notebook | KB text fields |
| Data sparsity metrics | Available through token statistics | KB text fields |

## Objective 2: Topic Classification

### Goal

Prove the chatbot understands what the user is asking about.

### Implementation

Topic routing is handled through:

- intent detection in `rag/retrieve.py`
- source-aware routing for HK official, business guide, YC examples, and sentiment questions
- inferred topic labels from retrieved chunks
- offline evaluation in `evaluation/run_evaluation.py`

The intent detector was extended with Hong Kong official phrases such as:

```text
annual return
business registration
company information
change company name
defunct solvent company
deregister
unique business identifier
```

### Data Evidence

Evaluation files:

```text
evaluation/test_queries.csv
data/evaluation/classification_report.csv
data/evaluation/confusion_matrix.csv
data/evaluation/summary.json
```

Latest result:

```text
Topic Accuracy: 100.0%
Weighted F1: 100.0%
```

### Charts and Tables Available

| Requested Evidence | Status | Data Source |
| --- | --- | --- |
| Confusion matrix | Available | `data/evaluation/confusion_matrix.csv` |
| Classification report | Available | `data/evaluation/classification_report.csv` |
| Weighted F1 score | Available | `data/evaluation/summary.json` |
| t-SNE / UMAP embedding plot | Data available, chart script can be added | Chroma embeddings / KB chunks |

## Objective 3: Information Retrieval and Ranking

### Goal

Demonstrate that search retrieves the most relevant information chunks.

### Implementation

Retrieval uses:

- `sentence-transformers` embedding model
- ChromaDB vector store in `data/chroma_db/`
- source-aware reranking in `rag/retrieve.py`
- Top-K result rendering in the Streamlit chatbot
- offline ranking evaluation in `evaluation/run_evaluation.py`

### Data Evidence

Evaluation output:

```text
data/evaluation/evaluation_results.csv
data/evaluation/topk_retrieval_samples.csv
data/evaluation/summary.json
```

Latest retrieval metrics:

```text
Source Recall@1: 100.0%
Source Recall@3: 100.0%
Source Recall@5: 100.0%
Relevant Recall@1: 83.33%
Relevant Recall@3: 94.44%
Relevant Recall@5: 94.44%
Relevant MRR: 0.8889
Relevant NDCG@5: 0.9034
```

### Charts and Tables Available

| Requested Evidence | Status | Data Source |
| --- | --- | --- |
| Search relevance histogram | Available with retrieved scores | `evaluation_results.csv` / retrieval output |
| Recall@K curve | Available | `summary.json`, `evaluation_results.csv` |
| Top-K retrieval sample table | Available | `topk_retrieval_samples.csv` |
| MRR / NDCG table | Available | `summary.json` |

## Objective 4: Answer Summarization

### Goal

Validate that chatbot answers are accurate, concise, and source-supported.

### Implementation

Answer generation is implemented in `rag/generate.py` using:

- retrieved source chunks
- a grounded RAG prompt
- Ollama generation
- follow-up generation after successful answers

Generated-answer evaluation is available through:

```text
evaluation/run_objective4.cmd
evaluation/run_evaluation.py --with-generation
```

### Data Evidence

Generated-answer output:

```text
data/evaluation/summary.json
data/evaluation/evaluation_results.csv
data/evaluation/topk_retrieval_samples.csv
```

Latest Objective 4 metrics:

```text
Answer metric mode: generated_answer
Generated answers: 22/22
ROUGE-1: 0.1446
ROUGE-2: 0.0406
ROUGE-L: 0.1104
```

### Charts and Tables Available

| Requested Evidence | Status | Data Source |
| --- | --- | --- |
| Source attribution pie chart | Available | retrieved sources in evaluation/chat history |
| Human evaluation radar chart | Structure available, needs manual ratings | can add `human_ratings.csv` |
| Raw retrieved text vs summarized output | Available | `topk_retrieval_samples.csv`, `evaluation_results.csv` |
| ROUGE score table | Available | `summary.json` |
| BERTScore / METEOR | Not implemented yet, can be added if required | generated answers + gold answers |

## Objective 5: Follow-Up Question Prediction

### Goal

Show how the chatbot anticipates the entrepreneur's next step.

### Implementation

Follow-up prediction is implemented in `rag/generate.py` and measured in `app.py`.

The Streamlit app now:

- renders generated follow-ups as clickable buttons
- logs every suggestion impression as `shown`
- logs every click as `clicked`
- queues clicked follow-ups as the next user query
- displays follow-up analytics in the dashboard

CMD summary helper:

```text
evaluation/run_objective5.cmd
```

### Data Evidence

Runtime log and summary files:

```text
data/analytics/followup_events.jsonl
data/analytics/followup_summary.json
data/analytics/followup_journeys.csv
```

Latest Objective 5 summary:

```text
Logged events: 18
Suggestions shown: 15
Suggestions clicked: 3
Follow-up CTR: 20.0%
Suggestion 1 CTR: 20.0%
Suggestion 2 CTR: 40.0%
Suggestion 3 CTR: 0.0%
```

Each follow-up event records:

```text
timestamp
session_id
event_type
turn_id
query
followup_text
followup_rank
topic
```

### Charts and Tables Available

| Requested Evidence | Status | Data Source |
| --- | --- | --- |
| User journey Sankey / flow table | Available after clicks | `followup_journeys.csv` |
| Prediction accuracy / CTR bar chart | Available after clicks | `followup_summary.json`, dashboard |
| CTR on suggested questions | Available | `followup_summary.json` |
| Perplexity | Not implemented; lower priority for current RAG follow-up workflow | would require LM probability scoring |

The current repository includes a small interaction log with real shown/clicked events. More user sessions will make the CTR and journey-flow chart more representative.

## Objective 6: Integrated Dashboard and Chatbot Analytics

### Goal

Provide a command-center view of system health, data coverage, retrieval behavior, sentiment signals, and evaluation metrics.

### Implementation

The Streamlit app has a dedicated `Analytics Dashboard` tab in `app.py`.

Current dashboard sections:

- overview metrics
- evaluation results
- follow-up prediction analytics
- knowledge source distribution
- top knowledge-base topics
- sentiment label distribution
- platform by sentiment
- average engagement by platform
- current session retrieval mix
- keyword profiles
- key insights
- diagnostic notes

### Data Evidence

Dashboard reads from:

```text
data/kb/summary.json
data/kb/cleaned_company_kb.jsonl
data/kb/*_chunks.jsonl
data/sentiment/sentiment_summary.json
data/sentiment/sentiment_cleaned.jsonl
data/sentiment/sentiment_keyword_profiles.json
data/evaluation/summary.json
data/evaluation/classification_report.csv
data/evaluation/topk_retrieval_samples.csv
data/analytics/followup_summary.json
data/analytics/followup_journeys.csv
```

### Charts and Tables Available

| Requested Evidence | Status | Data Source |
| --- | --- | --- |
| Real-time pulse / response latency | Partially available | evaluation latency and session history |
| Sentiment heatmap / grid | Available | sentiment platform/sentiment data |
| Interactive topic map / trending topics | Available as topic distribution and session retrieval mix | KB metadata and chat history |
| System health gauges | Partially available | dashboard metrics and app health checks |
| API uptime | Not long-term monitored yet | would need persistent uptime logs |

## How to Reproduce the Evidence

### Objective 2-4 Evaluation

Run from CMD:

```cmd
.venv\Scripts\python.exe evaluation\run_evaluation.py
```

For generated-answer Objective 4 metrics:

```cmd
evaluation\run_objective4.cmd
```

### Objective 5 Follow-Up Analytics

Open the app, ask a question, click suggested follow-up buttons, then run:

```cmd
evaluation\run_objective5.cmd
```

### Run the Dashboard

```cmd
.venv\Scripts\python.exe -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Summary

All six objectives are implementable in the current project.

| Objective | Current Status |
| --- | --- |
| Objective 1 | Implemented with KB data and keyword evidence |
| Objective 2 | Implemented with classification metrics |
| Objective 3 | Implemented with retrieval ranking metrics |
| Objective 4 | Implemented with generated-answer evaluation |
| Objective 5 | Implemented with click logging and CTR analytics |
| Objective 6 | Implemented with dashboard views and analytics integration |

Some charts require real interaction data to become meaningful, especially Objective 5 CTR and journey flow. The logging and dashboard infrastructure for those charts is already implemented.
