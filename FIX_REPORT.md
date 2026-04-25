# Fix Report

## Overview

This document summarizes the code fixes applied to the Entrepreneurship Chatbot project. The work focused on improving runtime stability, cleaning user-facing text, removing noisy debug output, and fixing a broken command-line smoke test.

## Files Changed

### `app.py`

- Replaced corrupted UI text with clean English labels.
- Removed runtime debug `print()` statements from the Streamlit chat flow.
- Kept the existing Streamlit workflow intact:
  - query rewriting
  - dense retrieval
  - source rendering
  - optional sentiment analysis
  - streaming answer generation
  - follow-up question generation
- Updated warning handling to use a clean `Warning:` prefix instead of corrupted characters.
- Improved source labels and sentiment panel labels for readable display.

### `rag/retrieve.py`

- Removed intent-detection debug output that printed on every retrieval call.
- Preserved existing intent detection, ChromaDB retrieval, source-aware reranking, and topic inference behavior.

### `rag/generate.py`

- Fixed the CLI smoke test at the bottom of the file.
- Corrected the return-value handling for `retrieve()`, which returns:

```python
matches, intent = retrieve(query)
```

- Added topic inference through `infer_topic_label(intent, matches)`.
- Replaced corrupted warning text in Ollama connection/request errors with clean English messages.
- Replaced the corrupted CLI separator with a plain ASCII separator.

### `sentiment/engine.py`

- Corrected the missing-assets error message.
- Old path:

```text
python sentiment_assets/build_sentiment_assets.py
```

- New path:

```text
python sentiment/build_assets.py
```

## Verification Performed

The following checks were run successfully:

```text
python -m py_compile app.py rag/retrieve.py rag/generate.py sentiment/engine.py
```

Dense retrieval smoke test:

```text
Query: How do I register a company in Hong Kong?
Result: 3 matches
Topic: HK Company Registration & Compliance
Sources: hk_official
```

Generation smoke test:

```text
python rag/generate.py "How do I register a company in Hong Kong?"
```

The command completed successfully and produced an answer using retrieved Hong Kong official guidance.

Streamlit check:

```text
HTTP 200
Local URL: http://localhost:8503
```

## Notes

- Ports `8501` and `8502` were already occupied by existing Python Streamlit processes, so the verified test instance was started on port `8503`.
- The project still depends on a running Ollama service and the configured model `qwen2.5:3b`.
- The current D-drive virtual environment path used by `run.bat` is:

```text
D:\python-envs\chatbot\Scripts\python.exe
```

## Result

The main chatbot application, retrieval path, and generation smoke test now run cleanly. The user-facing interface no longer displays corrupted UI labels in the repaired `app.py`, and unnecessary debug logs have been removed from the primary runtime path.

## Follow-Up Update: Analytics Dashboard Integration

After the initial fixes, an additional data-analysis and visualization module was implemented in the current project codebase. This work adapts the Analytics Dashboard concept identified in the reference GitHub project and maps it to this repository's local data layout.

### Updated File

#### `app.py`

- Added a two-tab Streamlit layout:

```text
Chatbot
Analytics Dashboard
```

- Kept the existing chatbot flow intact.
- Added an `Analytics Dashboard` tab that summarizes the project data and runtime session.
- Adapted all dashboard data paths to the current local project structure:

```text
data/kb/
data/sentiment/
```

### Added Analytics Functions

The following dashboard support functions were added:

```python
build_source_family_frame()
build_topic_distribution_frame()
build_sentiment_distribution_frame()
build_sentiment_platform_frame()
build_engagement_frame()
build_keyword_profile_frame()
build_session_retrieval_mix_frame()
render_horizontal_bar_chart()
render_grouped_sentiment_chart()
render_engagement_scatter()
render_overview_metrics()
render_key_insights()
render_analytics_dashboard()
```

### Dashboard Content Added

The dashboard now includes:

- Overview metrics:
  - Companies
  - Indexed Chunks
  - Topics
  - Sentiment Rows
  - Chat Turns
- Knowledge Source Distribution
- Top Knowledge-Base Topics
- Sentiment Label Distribution
- Platform by Sentiment
- Average Engagement by Platform
- Current Session Retrieval Mix
- Top Positive Keywords
- Top Neutral Keywords
- Key Insights
- Diagnostic Notes

### Data Sources Used

The dashboard reads from existing project outputs:

```text
data/kb/summary.json
data/kb/cleaned_company_kb.jsonl
data/kb/knowledge_base_chunks.jsonl
data/kb/hk_official_kb_chunks.jsonl
data/kb/business_guide_kb_chunks.jsonl
data/sentiment/sentiment_summary.json
data/sentiment/sentiment_cleaned.jsonl
data/sentiment/sentiment_keyword_profiles.json
```

### Visualization Technology

The dashboard uses:

```text
Streamlit
pandas
Vega-Lite via st.vega_lite_chart()
```

No Plotly or Matplotlib dependency was added.

### Verification Performed

The updated application was checked with:

```text
python -m py_compile app.py
```

The Streamlit application was then started successfully and returned:

```text
HTTP 200
Local URL: http://localhost:8504
```

### Result

The current project now includes both the repaired chatbot interface and a working data-analysis dashboard. The dashboard provides project-scale metrics, source distribution, topic coverage, sentiment analysis summaries, engagement statistics, keyword profiles, and live session retrieval mix visualization.

## Follow-Up Update: Offline Evaluation Evidence

An additional evaluation layer was added to support the six project objectives with evidence beyond the live demo. The goal of this update is to make topic routing, retrieval quality, and answer support measurable from a repeatable offline test set.

### Files Added

#### `evaluation/test_queries.csv`

- Added 22 manually defined evaluation queries.
- Covered four major objective areas:
  - Hong Kong company registration and compliance
  - Business planning and retail operations
  - YC startup pattern retrieval
  - Social media sentiment questions
- Each row includes:

```text
query
expected_topic_group
expected_source_family
relevant_id
gold_answer
expected_followup_type
```

#### `evaluation/run_evaluation.py`

- Added a reusable offline evaluation runner.
- Evaluates:
  - Topic classification / routing accuracy
  - Classification report and confusion matrix
  - Source-family retrieval accuracy
  - Relevant result Recall@K
  - Mean Reciprocal Rank (MRR)
  - NDCG@5
  - ROUGE-style answer-support coverage
  - Average retrieval latency
- The default mode avoids Ollama calls and compares retrieved context against gold answers.
- An optional generation mode is available:

```text
python evaluation/run_evaluation.py --with-generation
```

### Files Generated

Running the evaluation script produces:

```text
data/evaluation/summary.json
data/evaluation/evaluation_results.csv
data/evaluation/classification_report.csv
data/evaluation/confusion_matrix.csv
data/evaluation/topk_retrieval_samples.csv
```

### Files Updated

#### `rag/retrieve.py`

- Added official Hong Kong phrase detection for queries that do not contain the original single-word official terms.
- Added phrase coverage for cases such as:

```text
annual return
business registration
company information
change company name
defunct solvent company
deregister
unique business identifier
```

- This improved classification of Hong Kong compliance queries such as annual returns, deregistration, and company-name changes.

#### `app.py`

- Added an `Evaluation Results` section to the Analytics Dashboard.
- The dashboard now reads from:

```text
data/evaluation/summary.json
data/evaluation/classification_report.csv
data/evaluation/topk_retrieval_samples.csv
```

- Added dashboard metrics for:
  - Topic Accuracy
  - Weighted F1
  - Recall@5
  - MRR
  - Average Latency
  - Source Recall@K
  - Relevant Recall@K
  - ROUGE-style coverage

### Latest Evaluation Results

The offline evaluation was run with the project virtual environment:

```text
.\.venv\Scripts\python.exe evaluation\run_evaluation.py
```

The latest baseline summary was:

```text
Test queries: 22
Topic Accuracy: 100.0%
Weighted F1: 100.0%
Source Recall@1: 100.0%
Source Recall@3: 100.0%
Source Recall@5: 100.0%
Relevant Recall@1: 83.33%
Relevant Recall@3: 94.44%
Relevant Recall@5: 94.44%
Source MRR: 1.0000
Relevant MRR: 0.8889
Relevant NDCG@5: 0.9034
Average latency: 632.35 ms
Answer metric mode: retrieved_context
```

### Objective 4 Generated-Answer Evaluation

The answer summarization objective was then evaluated with live Ollama generation enabled:

```text
.\.venv\Scripts\python.exe evaluation\run_evaluation.py --with-generation
```

For CMD users, a helper script was added:

```text
evaluation\run_objective4.cmd
```

The generated-answer evaluation produced:

```text
Test queries: 22
Answer metric mode: generated_answer
Generated answers: 22/22
ROUGE-1: 0.1446
ROUGE-2: 0.0406
ROUGE-L: 0.1104
Topic Accuracy: 100.0%
Weighted F1: 100.0%
Relevant Recall@5: 94.44%
Relevant MRR: 0.8889
Relevant NDCG@5: 0.9034
Average retrieval latency: 605.32 ms
```

This upgrades Objective 4 from a retrieved-context support check to a real generated-answer evaluation. The per-query generated answer previews and gold-answer comparisons are available in:

```text
data/evaluation/evaluation_results.csv
data/evaluation/topk_retrieval_samples.csv
```

### Verification Performed

Code compilation was checked with:

```text
.\.venv\Scripts\python.exe -m py_compile app.py rag\retrieve.py evaluation\run_evaluation.py
```

The Streamlit application was started and checked successfully:

```text
HTTP 200
Local URL: http://localhost:8501
```

### Notes

- The default evaluation run still avoids Ollama and uses retrieved-context support metrics.
- To refresh actual generated chatbot answer metrics, start Ollama and run:

```text
.\.venv\Scripts\python.exe evaluation\run_evaluation.py --with-generation
```

- CMD users can also run:

```text
evaluation\run_objective4.cmd
```

- The project directory is not currently a Git repository, so `git status` could not be used for change tracking.

### Result

The project now has a repeatable offline evaluation workflow. Objectives 2 and 3 are directly supported by measurable classification and retrieval metrics, while Objective 4 now has both a baseline retrieved-context support metric and a generated-answer evaluation path.

## Follow-Up Update: Objective 5 Follow-Up Prediction Analytics

Objective 5 was implemented as a measurable follow-up suggestion workflow. Suggested follow-up questions are now clickable, clicks are logged, and the Analytics Dashboard reports follow-up click-through metrics.

### Files Added

#### `evaluation/summarize_followups.py`

- Reads follow-up impression and click logs.
- Calculates:
  - suggestions shown
  - suggestions clicked
  - overall CTR
  - CTR by suggestion rank
  - query-to-follow-up journey transitions
- Writes:

```text
data/analytics/followup_summary.json
data/analytics/followup_journeys.csv
```

#### `evaluation/run_objective5.cmd`

- CMD helper for Objective 5 analytics.
- Run from the project root with:

```text
evaluation\run_objective5.cmd
```

### Files Updated

#### `app.py`

- Replaced static suggested follow-up text with clickable Streamlit buttons.
- Clicking a suggested follow-up now:
  - logs a `clicked` event
  - queues the follow-up as the next user query
  - continues the conversation automatically
- Generated follow-up suggestions now log `shown` events.
- Added persistent per-session IDs and per-turn IDs.
- Added follow-up analytics to the dashboard:
  - Suggestions Shown
  - Suggestions Clicked
  - Follow-Up CTR
  - Logged Events
  - CTR by suggestion rank
  - journey transition table

### Runtime Log Files

The app now writes follow-up interaction evidence to:

```text
data/analytics/followup_events.jsonl
```

Each event includes:

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

### Verification Performed

Compilation checks passed:

```text
.\.venv\Scripts\python.exe -m py_compile app.py evaluation\summarize_followups.py
```

Objective 5 CMD helper ran successfully:

```text
cmd /c evaluation\run_objective5.cmd
```

Latest summary after a small interaction run:

```text
events: 18
suggestions_shown: 15
suggestions_clicked: 3
ctr: 0.2
suggestion_1_ctr: 0.2
suggestion_2_ctr: 0.4
suggestion_3_ctr: 0.0
```

Streamlit health check:

```text
HTTP 200
Local URL: http://localhost:8501
```

### Notes

- To refresh CTR evidence:
  1. Open the Streamlit app.
  2. Ask a question with follow-up generation enabled.
  3. Click one or more suggested follow-up buttons.
  4. Run:

```text
evaluation\run_objective5.cmd
```

### Result

Objective 5 now has a complete measurement path: generated suggestions, click logging, CTR calculation, journey transition output, and dashboard visualization.
