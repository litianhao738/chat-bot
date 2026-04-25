# Entrepreneurship Chatbot Pro

ARIN 7102 Project 8: Online Entrepreneurship Education Chatbot.

This branch, `final-chat-bot`, is the evaluation-ready version of the project. It includes:

- a YC startup knowledge base
- Hong Kong Companies Registry official guidance
- business guide knowledge chunks
- social media sentiment assets
- dense retrieval with ChromaDB and `sentence-transformers`
- local answer generation through Ollama
- follow-up question prediction
- Streamlit chatbot and analytics dashboard
- Objective 1-6 evaluation evidence and documentation

Detailed objective evidence is documented here:

```text
docs/OBJECTIVES_1_6_IMPLEMENTATION.md
```

## 1. System Requirements

Recommended environment:

- Windows 10/11
- Python 3.11+
- CMD terminal
- Ollama for local LLM generation

The project has also been tested with newer Python versions in the local development environment.

## 2. Clone the Repository

```cmd
git clone -b final-chat-bot https://github.com/litianhao738/chat-bot.git
cd chat-bot
```

If you already cloned the repository:

```cmd
git checkout final-chat-bot
git pull origin final-chat-bot
```

## 3. Create and Activate a Virtual Environment

Create a local virtual environment:

```cmd
python -m venv .venv
```

Activate it in CMD:

```cmd
.venv\Scripts\activate
```

Upgrade `pip`:

```cmd
python -m pip install --upgrade pip
```

Install Python packages:

```cmd
python -m pip install -r requirements.txt
```

Current runtime dependencies are listed in:

```text
requirements.txt
```

Main packages:

- `streamlit`
- `chromadb`
- `sentence-transformers`
- `scikit-learn`
- `pandas`
- `requests`
- `vaderSentiment`

## 4. Install and Prepare Ollama

Ollama is used for local answer generation and follow-up question generation.

### Option A: Install with Winget

```cmd
winget install Ollama.Ollama
```

### Option B: Install from the Official Website

Download Ollama for Windows:

```text
https://ollama.com/download/windows
```

### Start Ollama

Usually Ollama starts automatically after installation. If needed, run:

```cmd
ollama serve
```

Keep that terminal open if Ollama is not running as a background service.

### Pull the Model

The default model used by this project is:

```text
qwen2.5:3b
```

Download it:

```cmd
ollama pull qwen2.5:3b
```

Check that Ollama can see the model:

```cmd
ollama list
```

Check the local Ollama API:

```cmd
curl http://localhost:11434/api/tags
```

Optional: use a different Ollama model by setting `OLLAMA_MODEL` before running the app:

```cmd
set OLLAMA_MODEL=qwen2.5:3b
```

The app also lets you change the model name in the Streamlit sidebar.

## 5. Run the Chatbot

### Recommended Quick Start

Use the included startup script:

```cmd
run.bat
```

The script will:

1. select the available Python environment
2. check Streamlit
3. check whether `data\chroma_db` exists
4. build the retrieval index if needed
5. launch Streamlit

### Manual Run

If the ChromaDB index already exists:

```cmd
python -m streamlit run app.py
```

If the index is missing or you want to rebuild it:

```cmd
python rag\build_index.py --reset
python -m streamlit run app.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## 6. Demo Questions

Use these questions to show the main functions.

### Hong Kong Official Guidance

```text
How do I register a company in Hong Kong?
What annual return does a local private company need to file in Hong Kong?
How do I change a Hong Kong company name?
```

### Business Planning and Retail Operations

```text
What should I plan before opening a retail store?
How should a retailer manage inventory and supply chain?
How do I choose a target market for a small business?
```

### YC Startup Patterns

```text
Show me YC examples of companies in payments or credit.
Which YC companies are building healthcare or patient-care products?
What YC patterns appear in developer tools or APIs?
```

### Social Media Sentiment

```text
What is the overall social media sentiment in the dataset?
What are people saying on Twitter and Instagram in the sentiment dataset?
Which hashtags are associated with positive social media sentiment?
```

### Suggested Presentation Flow

1. Ask a Hong Kong registration question.
2. Ask a business planning question.
3. Ask a YC startup examples question.
4. Ask a sentiment question.
5. Click one suggested follow-up question.
6. Open the Analytics Dashboard tab.

This demonstrates source routing, retrieval, answer generation, sentiment analytics, follow-up prediction, and dashboard evidence.

## 7. Run Objective Evaluation

Evaluation assets are stored in:

```text
evaluation/
data/evaluation/
data/analytics/
```

### Objective 2-4 Offline Evaluation

Run:

```cmd
python evaluation\run_evaluation.py
```

This generates:

```text
data/evaluation/summary.json
data/evaluation/evaluation_results.csv
data/evaluation/classification_report.csv
data/evaluation/confusion_matrix.csv
data/evaluation/topk_retrieval_samples.csv
```

### Objective 4 Generated-Answer Evaluation

Make sure Ollama is running, then run:

```cmd
evaluation\run_objective4.cmd
```

This calls:

```cmd
python evaluation\run_evaluation.py --with-generation
```

Latest generated-answer metrics include:

```text
Generated answers: 22/22
ROUGE-1: 0.1446
ROUGE-2: 0.0406
ROUGE-L: 0.1104
Relevant Recall@5: 94.44%
Relevant MRR: 0.8889
```

### Objective 5 Follow-Up Analytics

First interact with the app:

1. Ask a question.
2. Wait for suggested follow-ups.
3. Click one or more follow-up buttons.

Then run:

```cmd
evaluation\run_objective5.cmd
```

This generates or refreshes:

```text
data/analytics/followup_summary.json
data/analytics/followup_journeys.csv
```

The runtime event log is:

```text
data/analytics/followup_events.jsonl
```

Current example evidence:

```text
Suggestions shown: 15
Suggestions clicked: 3
Follow-up CTR: 20.0%
```

## 8. Objective 1-6 Evidence Map

| Objective | Evidence Location |
| --- | --- |
| Objective 1: Knowledge Base and Keyword Extraction | `data/kb/`, `data/sentiment/`, `global_keywords.csv`, `company_keywords.csv` |
| Objective 2: Topic Classification | `evaluation/test_queries.csv`, `data/evaluation/classification_report.csv`, `data/evaluation/confusion_matrix.csv` |
| Objective 3: Retrieval and Ranking | `data/evaluation/summary.json`, `data/evaluation/topk_retrieval_samples.csv` |
| Objective 4: Answer Summarization | `evaluation/run_objective4.cmd`, `data/evaluation/evaluation_results.csv` |
| Objective 5: Follow-Up Prediction | `data/analytics/followup_events.jsonl`, `followup_summary.json`, `followup_journeys.csv` |
| Objective 6: Dashboard and Analytics | `app.py`, Analytics Dashboard tab |

Full explanation:

```text
docs/OBJECTIVES_1_6_IMPLEMENTATION.md
```

## 9. Folder and File Guide

### Root Files

| Path | Purpose |
| --- | --- |
| `app.py` | Main Streamlit chatbot and analytics dashboard |
| `config.py` | Central paths, model names, source routing terms, retrieval constants |
| `requirements.txt` | Python runtime dependencies |
| `run.bat` | CMD startup helper for Windows |
| `FIX_REPORT.md` | Change log and implementation notes |
| `README.md` | This setup and usage guide |

### `builders/`

Scripts that build processed knowledge-base files from raw or external source data.

Important files:

```text
builders/build_yc_kb.py
builders/build_hk_official_kb.py
builders/build_business_guide_kb.py
builders/source_manifest.json
```

### `rag/`

Retrieval and answer-generation logic.

Important files:

```text
rag/build_index.py
rag/retrieve.py
rag/generate.py
```

Roles:

- build ChromaDB index
- detect query intent
- retrieve Top-K chunks
- rerank retrieved evidence
- build prompts
- stream answers from Ollama
- generate follow-up questions

### `sentiment/`

Social media sentiment processing and analysis.

Important files:

```text
sentiment/build_assets.py
sentiment/engine.py
sentiment/input/sentimentdataset.csv
```

Roles:

- clean the social media dataset
- calculate sentiment summaries
- build keyword profiles
- provide the Sentiment Lens in the app

### `scrapers/`

YC scraping and data repair scripts.

Important files:

```text
scrapers/yc_scraper.py
scrapers/yc_deep_scraper.py
scrapers/yc_page_extract.py
scrapers/repair_descriptions.py
```

### `data/kb/`

Processed knowledge-base outputs.

Important files:

```text
data/kb/cleaned_company_kb.jsonl
data/kb/knowledge_base_chunks.jsonl
data/kb/hk_official_kb_chunks.jsonl
data/kb/business_guide_kb_chunks.jsonl
data/kb/global_keywords.csv
data/kb/company_keywords.csv
data/kb/topic_taxonomy.json
data/kb/summary.json
```

### `data/chroma_db/`

Persistent ChromaDB vector index used by runtime retrieval.

If this folder is missing, run:

```cmd
python rag\build_index.py
```

### `data/sentiment/`

Processed sentiment outputs used by the dashboard and Sentiment Lens.

Important files:

```text
data/sentiment/sentiment_cleaned.jsonl
data/sentiment/sentiment_summary.json
data/sentiment/sentiment_keyword_profiles.json
```

### `data/evaluation/`

Offline evaluation results for Objectives 2-4.

Important files:

```text
data/evaluation/summary.json
data/evaluation/classification_report.csv
data/evaluation/confusion_matrix.csv
data/evaluation/evaluation_results.csv
data/evaluation/topk_retrieval_samples.csv
```

### `data/analytics/`

Runtime analytics for Objective 5 follow-up prediction.

Important files:

```text
data/analytics/followup_events.jsonl
data/analytics/followup_summary.json
data/analytics/followup_journeys.csv
```

### `evaluation/`

Evaluation scripts and CMD helpers.

Important files:

```text
evaluation/test_queries.csv
evaluation/run_evaluation.py
evaluation/run_objective4.cmd
evaluation/summarize_followups.py
evaluation/run_objective5.cmd
```

### `docs/`

Project documentation.

Important file:

```text
docs/OBJECTIVES_1_6_IMPLEMENTATION.md
```

## 10. Troubleshooting

### Streamlit Is Missing

```cmd
python -m pip install -r requirements.txt
```

### Ollama Is Not Reachable

Check:

```cmd
ollama list
curl http://localhost:11434/api/tags
```

If needed:

```cmd
ollama serve
```

### Model Is Missing

```cmd
ollama pull qwen2.5:3b
```

### ChromaDB Index Is Missing

```cmd
python rag\build_index.py
```

Or rebuild from scratch:

```cmd
python rag\build_index.py --reset
```

### Objective 4 Evaluation Gives Fallback Results

That means Ollama was not reachable or generation failed.

Start Ollama and rerun:

```cmd
evaluation\run_objective4.cmd
```

### Objective 5 CTR Is Zero

CTR is based on real clicks. Ask a question in the app, click suggested follow-ups, then rerun:

```cmd
evaluation\run_objective5.cmd
```

## 11. Current Status

All six objectives are implemented or have an implemented measurement path:

| Objective | Status |
| --- | --- |
| Objective 1 | Implemented with KB and keyword data |
| Objective 2 | Implemented with classification report and confusion matrix |
| Objective 3 | Implemented with Recall@K, MRR, NDCG, and Top-K samples |
| Objective 4 | Implemented with generated-answer ROUGE-style evaluation |
| Objective 5 | Implemented with clickable follow-ups and CTR logging |
| Objective 6 | Implemented in the Streamlit Analytics Dashboard |

The most detailed evidence is in:

```text
docs/OBJECTIVES_1_6_IMPLEMENTATION.md
```
