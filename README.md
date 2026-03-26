# Entrepreneurship Chatbot

ARIN 7102 Project 8: Online Entrepreneurship Education Chatbot.

This project builds a multi-source entrepreneurship chatbot with:
- a YC startup knowledge base
- a Hong Kong official guidance sub-KB
- a business guide sub-KB
- a lightweight inquiry-topic classifier
- retrieval, structured answering, and follow-up generation
- a supplementary social-media sentiment module
- a Streamlit demo with chatbot and analytics dashboard

## Installation

Use Python 3.11+ if possible. In the current Windows setup, the project is also running on Python 3.14.

Install dependencies:

```cmd
cd /d "Your path"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m playwright install chromium
```

Optional local 3B answer polishing through Ollama:

This part is optional. If you do not need local answer polishing, you can skip Ollama and the chatbot will still run with the structured answer pipeline.

If you want local polishing, follow the steps below on Windows.

1. Download and install Ollama.
Official download page:

`https://ollama.com/download/windows`

You can install it with `winget`:

```cmd
winget install Ollama.Ollama
```

2. Find your `ollama.exe`, then set `OLLAMA_PATH` manually.

The project uses `OLLAMA_PATH` to locate Ollama. Replace the sample path below if your installation folder is different:

```cmd
set OLLAMA_PATH=C:\Users\%USERNAME%\AppData\Local\Programs\Ollama\ollama.exe
```

3. If your Ollama models are stored in a custom folder, set `OLLAMA_MODELS` as well.

This tells Ollama where the downloaded local model files are stored:

```cmd
set OLLAMA_MODELS=D:\ollama\models
```

4. Check that Ollama is reachable, then download the model used by this project.

```cmd
"%OLLAMA_PATH%" --version
"%OLLAMA_PATH%" list
"%OLLAMA_PATH%" pull qwen2.5:3b
```

5. Test local polishing from the project folder.

```cmd
python rag\generate_answer.py --query "How can I start a healthcare startup in Hong Kong?" --polish-with-local-llm
```

6. Start the demo in the same terminal session.

```cmd
python -m streamlit run app.py
```

Example from the current machine:

- `ollama.exe` is installed at `C:\Users\litia\AppData\Local\Programs\Ollama\ollama.exe`
- the local model files are stored at `D:\ollama\models`

Example commands for this machine:

```cmd
set OLLAMA_PATH=C:\Users\litia\AppData\Local\Programs\Ollama\ollama.exe
set OLLAMA_MODELS=D:\ollama\models
python -m streamlit run app.py
```

Important note:

- this project uses `OLLAMA_PATH` to locate `ollama.exe`
- the code does not scan the whole computer for Ollama
- the default local model is `qwen2.5:3b`
- if you want to use a different local Ollama model, change the model name in the Streamlit sidebar or set `OLLAMA_MODEL` before running the project
- the `set` commands above apply to the current terminal session; if you open a new terminal, set them again before running the app

Optional sentiment dataset input:

- place `sentimentdataset.csv` under `sentiment_assets/input/`
- the builder now uses `sentiment_assets/input/sentimentdataset.csv` as the default dataset path

## Run

### Fastest way to run the demo

If the processed data already exists, the minimum steps are:

```cmd
cd /d "Your path"
python rag\build_retrieval_index.py
python -m streamlit run app.py
```

### Recommended full build sequence

Run this if you want to rebuild the project outputs from the current repository:

```cmd
cd /d "Your path"
python build_keywords.py
python external_sources\build_hk_official_kb.py
python external_sources\build_business_guide_kb.py
python sentiment_assets\build_sentiment_assets.py
python classification\classify_inquiry_topics.py
python rag\build_retrieval_index.py
python -m streamlit run app.py
```

### Optional checks

Local answer polishing test:

```cmd
python rag\generate_answer.py --query "How can I start a healthcare startup in Hong Kong?" --polish-with-local-llm
```

Sentiment engine test:

```cmd
python sentiment_assets\sentiment_engine.py --text "What are customers complaining about most in online retail reviews?"
```

## Demo Questions

Use these during presentation or grading.

### Hong Kong official guidance

- `How do I register a company in Hong Kong?`
- `What official steps are needed to change a company name in Hong Kong?`

### Business guide / entrepreneurship advice

- `How do I open a retail store and plan pricing for target customers?`
- `How should I think about competition and customer targeting for a new online store?`
- `How do I start a business and validate a market idea?`

### YC startup pattern questions

- `How can I start a healthcare startup in Hong Kong?`
- `What YC startup patterns are relevant for a B2B SaaS founder?`

### Sentiment questions

These should auto-trigger the `Sentiment Lens`:

- `What are customers complaining about most in online retail reviews?`
- `Summarize social media sentiment about food delivery service delays.`
- `What negative feedback do users give about fintech apps?`

### Good presentation order

Ask these in sequence to show all major capabilities:

1. `How do I register a company in Hong Kong?`
2. `How do I open a retail store and plan pricing for target customers?`
3. `What are customers complaining about most in online retail reviews?`
4. `How can I start a healthcare startup in Hong Kong?`

This sequence helps demonstrate:
- `hk_official` routing
- `business_guide` routing
- `sentiment_assets` triggering
- YC startup-pattern retrieval
- the live `Current Session Retrieval Mix` chart in the dashboard

## Modules

### 1. YC Knowledge Base

Main purpose:
- build the core entrepreneurship knowledge base from YC company pages

Main scripts:
- `logic.py`
- `logic_deep_scraper.py`
- `repair_yc_empty_descriptions.py`
- `build_keywords.py`
- `check_project8_kb_quality.py`

Main outputs:
- `project8_kb/cleaned_company_kb.jsonl`
- `project8_kb/company_keywords.csv`
- `project8_kb/global_keywords.csv`
- `project8_kb/topic_taxonomy.json`
- `project8_kb/knowledge_base_chunks.jsonl`
- `project8_kb/summary.json`
- `project8_kb/data_quality_report.json`

Current role:
- objective 1 keyword extraction and taxonomy building
- main startup example source for retrieval and answer generation

### 2. Inquiry Topic Classification

Main purpose:
- classify user enquiries into knowledge-base topics before retrieval

Main script:
- `classification/classify_inquiry_topics.py`

Main outputs:
- `classification/output/inquiry_topic_classifier.pkl`
- `classification/output/inquiry_topic_classifier_report.json`
- `classification/output/sample_inquiry_predictions.json`
- `classification/output/topic_catalog.json`

Current method:
- lightweight TF-IDF centroid classifier
- weak supervision from existing knowledge-base topic labels

Current role:
- objective 2 topic routing

### 3. Hong Kong Official Sub-KB

Main purpose:
- add official Hong Kong company-registration and compliance guidance

Main script:
- `external_sources/build_hk_official_kb.py`

Main source family:
- Companies Registry pages from `cr.gov.hk`

Main outputs:
- `external_sources/processed/hk_official_kb.jsonl`
- `external_sources/processed/hk_official_kb_chunks.jsonl`
- `external_sources/processed/hk_official_kb_summary.json`

Current role:
- prioritize official guidance for registration, compliance, and procedural questions

### 4. Business Guide Sub-KB

Main purpose:
- add broader entrepreneurship, retail, operations, pricing, and supply-chain guidance

Main script:
- `external_sources/build_business_guide_kb.py`

Current planned sources:
- `Business News Daily`
- `Forbes Advisor`
- `NetSuite`

Main outputs:
- `external_sources/processed/business_guide_kb.jsonl`
- `external_sources/processed/business_guide_kb_chunks.jsonl`
- `external_sources/processed/business_guide_kb_summary.json`

Current role:
- support how-to-start, retail, pricing, competition, target-customer, and operations questions

Implementation note:
- the builder first tries `requests`
- if needed it falls back to `curl_cffi` browser impersonation

### 5. Retrieval, Answering, and Follow-ups

Main purpose:
- retrieve the best pages and produce a structured entrepreneurship answer

Main scripts:
- `rag/build_retrieval_index.py`
- `rag/retrieve_chunks.py`
- `rag/generate_answer.py`
- `rag/generate_followups.py`

Current retrieval behavior:
- unified multi-source index
- topic-aware routing
- source-aware scoring
- diversity rerank and dedup
- official Hong Kong priority for compliance questions
- business-guide priority for retail and operations questions
- YC priority for startup-pattern and case-style questions

Current answer behavior:
- structured answer sections
- evidence summary
- evidence gap detection
- source-aware wording
- optional local 3B polishing with Ollama

### 6. Sentiment Assets

Main purpose:
- provide a supplementary social-media text analysis component for Project 8

Main scripts:
- `sentiment_assets/build_sentiment_assets.py`
- `sentiment_assets/sentiment_engine.py`

Main outputs:
- `sentiment_assets/output/sentiment_cleaned.jsonl`
- `sentiment_assets/output/sentiment_summary.json`
- `sentiment_assets/output/sentiment_keyword_profiles.json`

Current role:
- separate from the main RAG pipeline
- powers the analytics dashboard and the per-question `Sentiment Lens`
- only auto-triggers for feedback, complaint, review, and social-media style questions

Important limitation:
- the uploaded Kaggle dataset is heavily neutral, so this module should be presented as supplementary evidence rather than the main decision engine

### 7. Streamlit Demo

Main file:
- `app.py`

Current UI features:
- multi-turn chatbot
- source-aware answer display
- follow-up questions
- optional local 3B polished answer
- analytics dashboard
- key insights panel
- live session retrieval mix
- sentiment lens for matching question types

## Current Core Outputs

These are the main files the final demo depends on:

- `project8_kb/knowledge_base_chunks.jsonl`
- `classification/output/inquiry_topic_classifier.pkl`
- `classification/output/topic_catalog.json`
- `external_sources/processed/hk_official_kb_chunks.jsonl`
- `external_sources/processed/business_guide_kb_chunks.jsonl`
- `rag/output/retrieval_index.pkl`
- `sentiment_assets/output/sentiment_summary.json`
- `sentiment_assets/output/sentiment_keyword_profiles.json`

## Current Status Snapshot

At the current stage, the system already supports:
- multi-source entrepreneurship question answering
- Hong Kong official guidance retrieval
- business-guide retrieval for retail and operations questions
- follow-up question prediction
- supplementary sentiment analysis
- chatbot plus analytics dashboard demo

This version is intended to be runnable, explainable, and presentation-friendly for ARIN 7102 Project 8 submission.
