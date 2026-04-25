# Entrepreneurship Chatbot Pro

ARIN 7102 Project 8: Online Entrepreneurship Education Chatbot.

This branch, `final-chat-bot`, is the final demo version of the project. It includes:

- a YC startup knowledge base
- Hong Kong Companies Registry official guidance
- business guide knowledge chunks
- social media sentiment assets
- dense retrieval with ChromaDB and `sentence-transformers`
- local answer generation through Ollama
- follow-up question prediction
- a Streamlit chatbot and analytics dashboard

## 1. System Requirements

Recommended environment:

- Windows 10/11
- Python 3.11+
- CMD terminal
- Ollama for local LLM generation

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

Install project packages:

```cmd
python -m pip install -r requirements.txt
```

Main dependencies:

- `streamlit`
- `chromadb`
- `sentence-transformers`
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `plotly`
- `kaleido`
- `requests`
- `vaderSentiment`

## 4. Install and Prepare Ollama

Ollama is used for local answer generation and follow-up question generation.

This project expects the Ollama server to be reachable on your machine. The default local model is:

```text
qwen2.5:3b
```

### 4.1 Download and Install Ollama

Official Windows download page:

```text
https://ollama.com/download/windows
```

You can also install Ollama with Winget:

```cmd
winget install Ollama.Ollama
```

After installation, check whether CMD can find `ollama`:

```cmd
ollama --version
```

If `ollama` is not recognized, find your `ollama.exe` and set `OLLAMA_PATH` manually for the current CMD session. Replace the sample path if your installation folder is different:

```cmd
set OLLAMA_PATH=%LOCALAPPDATA%\Programs\Ollama\ollama.exe
"%OLLAMA_PATH%" --version
```

Example from the current Windows machine:

```text
C:\Users\litia\AppData\Local\Programs\Ollama\ollama.exe
```

### 4.2 Optional: Set a Custom Model Folder

If your Ollama models are stored in a custom folder, set `OLLAMA_MODELS` before pulling or running models:

```cmd
set OLLAMA_MODELS=D:\ollama\models
```

If you use Ollama's default model folder, you can skip this step.

### 4.3 Start Ollama and Download the Model

Start Ollama if it is not already running:

```cmd
ollama serve
```

If you set `OLLAMA_PATH`, use:

```cmd
"%OLLAMA_PATH%" serve
```

Open another CMD window in the project folder, then download the default model. If `ollama` is not on PATH in the new CMD window, set `OLLAMA_PATH` there again first:

```cmd
ollama pull qwen2.5:3b
```

Or, if using `OLLAMA_PATH`:

```cmd
"%OLLAMA_PATH%" pull qwen2.5:3b
```

Check that the model is available:

```cmd
ollama list
```

Or:

```cmd
"%OLLAMA_PATH%" list
```

Finally, check the local Ollama API:

```cmd
curl http://localhost:11434/api/tags
```

### 4.4 Optional: Use a Different Ollama Model

Use another model by setting `OLLAMA_MODEL` before running the app:

```cmd
set OLLAMA_MODEL=qwen2.5:3b
```

The model name can also be changed in the Streamlit sidebar.

Important notes:

- `qwen2.5:3b` is the default local model used by this branch.
- `set` commands only apply to the current CMD session. If you open a new terminal, set them again.
- The Streamlit app talks to the local Ollama API. Make sure `ollama serve` is running before asking chatbot questions that require generated answers.

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

Suggested demo flow:

1. Ask a Hong Kong registration question.
2. Ask a business planning question.
3. Ask a YC startup examples question.
4. Ask a sentiment question.
5. Click one suggested follow-up question.
6. Open the Analytics Dashboard tab.

## 7. Folder and File Guide

### Root Files

| Path | Purpose |
| --- | --- |
| `app.py` | Main Streamlit chatbot and analytics dashboard |
| `config.py` | Central paths, model names, source routing terms, retrieval constants |
| `requirements.txt` | Python runtime dependencies |
| `run.bat` | CMD startup helper for Windows |
| `FIX_REPORT.md` | Change log and implementation notes |
| `README.md` | Setup and usage guide |

### `builders/`

Builds processed knowledge-base files from raw or external data.

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

Main roles:

- build the ChromaDB index
- detect query intent
- retrieve and rerank chunks
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

Main roles:

- clean the social media dataset
- build sentiment summaries
- build keyword profiles
- power the app's Sentiment Lens

### `scrapers/`

YC scraping and data repair scripts.

Important files:

```text
scrapers/yc_scraper.py
scrapers/yc_deep_scraper.py
scrapers/yc_page_extract.py
scrapers/repair_descriptions.py
```

### `external_sources/`

Processed summaries for external Hong Kong official and business guide sources.

Important files:

```text
external_sources/processed/hk_official_kb_summary.json
external_sources/processed/business_guide_kb_summary.json
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

### `data/analytics/`

Runtime analytics generated from follow-up suggestion interactions.

Important files:

```text
data/analytics/followup_events.jsonl
data/analytics/followup_summary.json
data/analytics/followup_journeys.csv
```

### `evaluation/` and `outputs/`

Offline evaluation scripts and generated chart evidence for Objectives 1-5.

Important files:

```text
evaluation/run_evaluation.py
evaluation/generate_obj1_figures.py
evaluation/generate_obj2_figures.py
evaluation/generate_obj3_figures.py
evaluation/generate_obj4_figures.py
evaluation/generate_obj5_figures.py
evaluation/seed_obj5_followup_events.py
outputs/obj1/
outputs/obj2/
outputs/obj3/
outputs/obj4/
outputs/obj5/
```

## 8. Troubleshooting

### Streamlit or Dependencies Are Missing

```cmd
python -m pip install -r requirements.txt
```

### Ollama Is Not Reachable

Check:

```cmd
ollama --version
ollama list
curl http://localhost:11434/api/tags
```

If needed:

```cmd
ollama serve
```

If `ollama` is not recognized in CMD, set `OLLAMA_PATH` and run Ollama through the full path:

```cmd
set OLLAMA_PATH=%LOCALAPPDATA%\Programs\Ollama\ollama.exe
"%OLLAMA_PATH%" list
"%OLLAMA_PATH%" serve
```

### Model Is Missing

```cmd
ollama pull qwen2.5:3b
```

Or:

```cmd
"%OLLAMA_PATH%" pull qwen2.5:3b
```

### ChromaDB Index Is Missing

```cmd
python rag\build_index.py
```

Or rebuild from scratch:

```cmd
python rag\build_index.py --reset
```
