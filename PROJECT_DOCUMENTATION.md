# Online Entrepreneurship Education Chatbot Project Documentation

## 1. Project Overview

This project is developed for **ARIN 7102 Project 8**.  
The project theme is an **Online Entrepreneurship Education Chatbot** that combines data mining, text analysis, retrieval, question answering, follow-up prediction, and dashboard visualization.

Important data-source note:

- the project uses a mix of **live web collection**, **manifest-configured web sources**, and **one uploaded local CSV file**
- the sentiment module does **not** directly scrape Facebook, Instagram, or Twitter
- instead, the platform labels in the sentiment dashboard come from the uploaded Kaggle CSV, which already contains a `Platform` column

The core idea of the project is to build a chatbot that does not answer only from one source.  
Instead, it integrates:

- a **YC startup knowledge base** for startup examples and startup patterns,
- a **Hong Kong official knowledge base** for registration and compliance procedures,
- a **business guide knowledge base** for startup planning, retail, pricing, and operations advice,
- a **sentiment-analysis module** as a supplementary text-analytics component,
- and a **Streamlit chatbot + analytics dashboard** for demonstration.

This makes the system more suitable for entrepreneurship education, because the user may ask very different types of questions:

- official procedural questions,
- startup-pattern questions,
- practical launch questions,
- customer feedback or sentiment questions.

The current system is designed as a **multi-source retrieval and analytics platform**, not just a simple FAQ chatbot.

---

## 2. Project Objectives and How the System Addresses Them

The project follows the formal objectives of Project 8.

### Objective 1

**Use advanced data mining models to extract keywords from internet resources or social-media resources and build a knowledge base.**

How the project addresses this:

- YC startup company pages were collected and cleaned
- company descriptions were normalized
- TF-IDF n-gram keywords were extracted
- topic clustering was built using dimensionality reduction and clustering
- a knowledge-base-ready chunk dataset was produced

### Objective 2

**Use classification and/or text analysis models to classify enquiry topics in the knowledge base.**

How the project addresses this:

- a lightweight inquiry topic classifier was trained
- the classifier uses weak supervision from the knowledge-base topic labels
- the classifier is used as a query router before retrieval

### Objective 3

**Use advanced data mining and text analysis models to retrieve and filter the most relevant pages.**

How the project addresses this:

- a unified retrieval index was built
- the retrieval system supports topic-aware and source-aware ranking
- different source families are prioritized differently depending on the user’s question
- diversity reranking and deduplication are applied

### Objective 4

**Analyze and summarize the text from the most relevant pages to generate answers.**

How the project addresses this:

- the answer layer generates structured answers from retrieved evidence
- it summarizes evidence rather than simply returning raw chunks
- it distinguishes between official guidance, business-guide advice, and startup examples
- it also detects evidence gaps

### Objective 5

**Use predictive models and/or text analysis models to predict follow-up customer questions.**

How the project addresses this:

- a follow-up generation module predicts likely next questions
- it uses topic, answer content, recommendation, and retrieved signals

### Objective 6

**Integrate the work into a chatbot and/or dashboard.**

How the project addresses this:

- a Streamlit application provides:
  - multi-turn chatbot interaction
  - structured answer display
  - retrieval evidence display
  - sentiment lens
  - analytics dashboard

---

## 3. High-Level System Architecture

The current project can be understood as six connected layers.

### Layer 1. Data acquisition and data preparation

This layer collects and prepares raw data from:

- YC company pages collected from the Y Combinator company directory
- Hong Kong Companies Registry webpages from `cr.gov.hk`
- business guide webpages configured in the project manifest
- one uploaded Kaggle sentiment dataset stored inside the project

### Layer 2. Knowledge-base construction

This layer transforms collected raw data into:

- structured page-level records
- cleaned text fields
- keyword lists
- topic labels
- chunk-level retrieval records

### Layer 3. Topic classification

This layer predicts the likely topic of a user query before retrieval.

### Layer 4. Multi-source retrieval

This layer retrieves evidence from a unified index that includes:

- YC
- Hong Kong official pages
- business guide pages

### Layer 5. Answer and follow-up generation

This layer:

- summarizes evidence
- produces a structured answer
- identifies evidence gaps
- predicts likely follow-up questions
- optionally uses a local 3B Ollama model to polish the final wording

### Layer 6. Front-end demo and analytics

This layer exposes the whole system through:

- a multi-turn chatbot UI
- an analytics dashboard

---

## 4. Main Data Sources

### 4.1 YC startup knowledge base

This is the largest and most important source family in the project.

Actual source:

- Y Combinator company directory and company profile pages from `ycombinator.com`

How the data was obtained:

- `logic.py` collects company links from YC batch pages
- `logic_deep_scraper.py` visits each company page and extracts structured startup information
- `repair_yc_empty_descriptions.py` revisits weak rows and repairs missing descriptions

What kind of data this source contributes:

- company names
- one-liners
- descriptions
- website links
- founder and company metadata
- startup example text for later keyword extraction and topic clustering

Main purpose:

- provide startup examples
- provide startup patterns
- support market, product, healthcare, AI, fintech, and other startup-related topics

Current scale:

- 5,494 company records
- 20 topic clusters in the original YC topic taxonomy

Key characteristics:

- rich startup descriptions
- broad domain coverage
- suitable for example-based answers and startup-pattern retrieval

Strength:

- large coverage and broad thematic range

Limitation:

- not a local regulatory source
- not a procedural guide source

### 4.2 Hong Kong official knowledge base

Main purpose:

- provide official Hong Kong business registration and compliance guidance

Actual source:

- Companies Registry official webpages from `cr.gov.hk`

How the data was obtained:

- page URLs are explicitly listed in `source_manifest.json`
- `build_hk_official_kb.py` fetches and normalizes those pages
- the builder converts them into page-level records and chunk-level retrieval records

Current official pages configured in the project include:

- Companies Registry home page
- register company
- obtain company information
- change company name
- deregister company
- electronic services
- e-services FAQ
- local company incorporation FAQ
- business registration and miscellaneous FAQ
- annual return for local private company

What kind of data this source contributes:

- official procedures
- filing rules
- compliance instructions
- company-maintenance guidance
- Hong Kong-specific evidence for local entrepreneurship questions

Current coverage includes:

- company registration
- company information search
- company name change
- deregistration
- e-services
- incorporation FAQ
- annual return compliance

Strength:

- official and trustworthy procedural information

Limitation:

- much smaller than the YC source family

### 4.3 Business guide knowledge base

Main purpose:

- provide practical startup and retail guidance beyond startup examples

Actual configured sources:

- NetSuite
- Business News Daily
- Forbes Advisor

Configured page URLs include:

- NetSuite retail supply chain article
- Business News Daily retail-store article
- Forbes Advisor how-to-start-a-business article

How the data was obtained:

- page URLs, aliases, fallback summaries, and seed keywords are defined in `source_manifest.json`
- `build_business_guide_kb.py` attempts to fetch live webpage content
- if a page is blocked, the builder uses the structured fallback summary stored in the manifest

This means the project uses a **hybrid source strategy** for this source family:

- live fetch when possible
- manifest fallback when a site blocks direct access

This should be stated clearly because not every business-guide record necessarily comes from a successful live scrape in every run.

Current business-guide roles:

- general startup guidance
- retail launch guidance
- pricing and competition guidance
- operations and supply-chain guidance

Strength:

- very useful for “how-to-start” and “how-to-operate” style questions

Limitation:

- small source family
- some pages may be blocked by website restrictions during scraping

### 4.4 Sentiment assets

Main purpose:

- support supplementary sentiment-oriented analysis
- enrich the project with social-media text analysis

Actual source:

- one uploaded Kaggle CSV file named `sentimentdataset.csv`

How the data was obtained:

- the dataset was manually added into the project under `sentiment_assets/input`
- `build_sentiment_assets.py` reads this local CSV file
- the script does not fetch live data from Facebook, Instagram, or Twitter

Why Facebook, Instagram, and Twitter appear in the dashboard:

- the original Kaggle CSV already contains a `Platform` column
- the values in that column include `Facebook`, `Instagram`, and `Twitter`
- the dashboard groups and visualizes rows by that existing platform field

What kind of data this source contributes:

- social-media text
- sentiment labels
- hashtags
- platform labels
- likes
- retweets
- country and timestamp metadata

Important clarification:

- the project does **not** maintain separate raw Facebook, Instagram, or Twitter export files
- the platform-level sentiment charts are derived from the single uploaded Kaggle CSV
- therefore the sentiment module is **dataset-based**, not **live platform scraping based**

Current role:

- complaint / review / feedback / social-media style questions can trigger the sentiment lens
- the sentiment module is used for dashboard analytics and lightweight query-level analysis

Strength:

- adds a text-analytics and dashboard component beyond the main RAG pipeline

Limitation:

- the current dataset is generic and heavily neutral
- it should be treated as supplementary evidence, not a strong standalone market-intelligence source

---

## 5. Core Project Modules

### 5.1 YC data collection and repair

Main files:

- `logic.py`
- `logic_deep_scraper.py`
- `repair_yc_empty_descriptions.py`

Responsibilities:

- collect YC company links
- scrape company pages
- extract structured startup information
- repair missing descriptions and weak rows

This stage is important because later keyword extraction and retrieval quality depend heavily on description quality.

### 5.2 Knowledge-base construction

Main file:

- `build_keywords.py`

Responsibilities:

- clean and normalize startup text
- extract TF-IDF keywords
- generate a global keyword catalog
- cluster companies into topic groups
- generate taxonomy labels
- export chunked KB records

This is one of the most important files in the project because it turns raw startup data into a reusable knowledge base.

### 5.3 Quality checking

Main file:

- `check_project8_kb_quality.py`

Responsibilities:

- check duplicate or suspicious records
- identify empty descriptions or weak text
- inspect keyword coverage
- inspect topic distributions

This makes the KB construction process more reliable and easier to justify in the final report.

### 5.4 Inquiry topic classification

Main file:

- `classify_inquiry_topics.py`

Responsibilities:

- build a lightweight topic classifier
- generate weakly supervised training samples from the KB
- evaluate classification quality
- export model artifacts and reports

This module is used as a routing model before retrieval.

### 5.5 Hong Kong official KB builder

Main file:

- `build_hk_official_kb.py`

Responsibilities:

- build a structured local-regulation sub-KB
- turn official pages into retrieval-ready chunks

### 5.6 Business guide KB builder

Main file:

- `build_business_guide_kb.py`

Responsibilities:

- build a structured general startup and retail guide sub-KB
- keep this source family separate from official HK guidance
- export page-level and chunk-level artifacts

### 5.7 Retrieval index construction

Main file:

- `build_retrieval_index.py`

Responsibilities:

- merge all KB families into one unified index
- preserve source metadata for later source-aware answering

### 5.8 Multi-source retrieval

Main file:

- `retrieve_chunks.py`

Responsibilities:

- rank evidence chunks
- support topic-aware retrieval
- support source-aware routing
- prioritize:
  - official HK pages for local procedural questions
  - business guide pages for practical startup and retail questions
  - YC examples for startup-pattern and example questions

### 5.9 Answer generation

Main file:

- `generate_answer.py`

Responsibilities:

- parse query intent
- summarize evidence
- distinguish source families in the final answer
- detect evidence gaps
- optionally call local Ollama polishing

### 5.10 Follow-up generation

Main file:

- `generate_followups.py`

Responsibilities:

- predict likely next questions based on the answer and its signals

### 5.11 Sentiment module

Main files:

- `build_sentiment_assets.py`
- `sentiment_engine.py`

Responsibilities:

- clean the uploaded Kaggle sentiment dataset
- build sentiment summary outputs
- generate keyword profiles
- run lightweight query-level sentiment analysis when triggered

### 5.12 Streamlit demo

Main file:

- `app.py`

Responsibilities:

- provide a multi-turn chatbot interface
- display structured answers and evidence
- display follow-up suggestions
- optionally display local 3B polished output
- display the analytics dashboard
- show current session retrieval mix

---

## 6. Data Collection, Cleansing, and Pre-processing

### 6.1 Data collection

The project collected data from both web sources and uploaded file sources.

Collected data includes:

- YC startup pages from `ycombinator.com`
- Hong Kong official webpages from `cr.gov.hk`
- business-guide webpages defined in the project manifest
- one uploaded Kaggle sentiment CSV stored locally inside the repository

This gives the system both:

- startup knowledge
- official procedural knowledge
- practical business advice
- social-media sentiment data

### 6.2 Data cleansing

The project performed several cleaning operations:

- filling missing YC descriptions
- repairing suspicious company names
- normalizing one-liners and descriptions
- cleaning HTML and noisy symbols
- removing low-value tokens and stopwords
- converting sentiment labels into normalized categories
- converting likes and retweets into numeric values

This is important because noisy data would directly hurt:

- keyword quality
- topic clustering quality
- classifier training quality
- retrieval relevance

### 6.3 Data pre-processing

The project also completed several important pre-processing steps:

- keyword extraction with TF-IDF n-grams
- topic construction using SVD and KMeans
- chunk generation for later retrieval
- weak-supervision sample generation for the classifier
- building a unified multi-source retrieval index
- building dashboard-ready sentiment summary files

In short, the project does not use raw data directly.  
It converts raw web data into structured analytical assets.

---

## 7. Classification and Topic Routing

The system includes a lightweight inquiry-topic classifier.

Current reported results:

- accuracy: 0.6282
- macro F1: 0.6555
- weighted F1: 0.6104

Best-performing topics include:

- `cancer / drug / drugs`
- `financial / investors / capital`
- `games / game / play`
- `healthcare / care / medical`

Weak topics include:

- `review / tests / run`
- `insurance / security / search`
- `code / api / alternative`

Interpretation:

- the classifier is not a final predictive engine
- it is a routing model
- for that role, it is already useful

Why this matters:

- the classifier helps the retrieval system begin from more relevant topic candidates
- that usually improves retrieval quality and answer coherence

---

## 8. Multi-source Retrieval Design

One of the strongest parts of the project is the retrieval design.

### 8.1 Unified index

The system merges all source families into one index.

Current reported index statistics:

- indexed docs: 5,572
- topic labels in unified index: 31

Source distribution:

- YC: 5,494
- Hong Kong official: 54
- business guide: 24

### 8.2 Retrieval logic

The retrieval module supports:

- keyword overlap
- topic-aware recall
- query expansion
- quality-aware scoring
- source-aware routing
- official-source prioritization
- diversity reranking
- deduplication

### 8.3 Why this design matters

The user does not always ask the same kind of entrepreneurship question.

Examples:

- a registration question should prefer official procedural evidence
- a retail launch question should prefer business-guide evidence
- a startup-example question should prefer YC evidence

This makes the chatbot much more useful than a plain keyword-search tool.

---

## 9. Answer Generation Design

The answer layer is designed as an evidence-grounded structured answer system.

The answer module currently performs:

- query intent parsing
- evidence summarization
- source-aware section writing
- evidence gap detection
- recommendation generation
- next-step suggestion

This means answers are not just copied from retrieved chunks.  
They are organized into a more useful educational response.

The system also distinguishes between different evidence types:

- Hong Kong official guidance
- business-guide advice
- YC startup patterns

This distinction is especially important for trust and interpretability.

For example:

- the system should not present business-guide advice as if it were official legal instruction
- it should not present YC startup examples as if they were regulatory rules

That source separation is one of the strengths of the current answer design.

---

## 10. Follow-up Prediction

The follow-up module makes the project more interactive and more educational.

Instead of stopping at one answer, the system predicts likely next questions.

This is useful because entrepreneurship support often happens in sequences, for example:

- first ask about startup direction
- then ask about registration
- then ask about pricing or customer targeting
- then ask about operations or customer feedback

The follow-up module helps turn the chatbot into a guided learning experience.

---

## 11. Sentiment Analysis Design

The project also includes a sentiment layer.

### 11.1 What it currently does

The sentiment system:

- cleans the uploaded Kaggle sentiment dataset
- generates sentiment summaries and keyword profiles
- supports a lightweight query-level sentiment lens
- triggers only when the user question is clearly about:
  - complaints
  - reviews
  - feedback
  - social media reactions

### 11.2 Why this is useful

This helps the project satisfy the course requirement to include text analysis and predictive / analytical elements beyond the core chatbot pipeline.

Source clarification:

- Facebook, Instagram, and Twitter are **platform labels inside the uploaded CSV**
- they are not separately scraped live by this project
- the dashboard and sentiment summaries are computed from that one local dataset

### 11.3 Current limitation

The sentiment dataset is highly neutral-heavy:

- 682 neutral
- 46 positive
- 4 negative

Therefore:

- sentiment results should be presented as supplementary
- the dashboard appropriately treats them as descriptive rather than decisive

---

## 12. Analytics Dashboard

The Analytics Dashboard is a major presentation component of the project.

It is important because it shows that the project is not only a chatbot, but also an analytics system.

### 12.1 Overview metrics

The dashboard summarizes:

- number of companies
- number of indexed docs
- number of topics
- classifier accuracy
- number of sentiment rows

This gives the audience an immediate understanding of project scale.

### 12.2 Knowledge Source Distribution

This chart shows the source-family balance in the unified index.

Key insight:

- the system is YC-heavy
- smaller sources are strategically important for official guidance and practical advice

### 12.3 Top Knowledge-Base Topics

This chart shows which topics are most represented in the KB.

Key insight:

- some startup themes have much stronger coverage than others

### 12.4 Classifier F1 Score by Topic

This chart visualizes per-topic classification performance.

Key insight:

- routing quality differs by topic
- some topics are easy to classify, some remain weak

### 12.5 Lowest-F1 diagnostic table

This table provides more precise diagnosis for weak topics.

Key insight:

- some topic labels need more training examples or clearer boundaries

### 12.6 Sentiment Label Distribution

This chart shows the class imbalance in the sentiment dataset.

Key insight:

- the sentiment dataset is useful, but clearly limited

### 12.7 Platform by Sentiment

This chart shows sentiment counts across Facebook, Instagram, and Twitter.

Key insight:

- neutral posts dominate all platforms

### 12.8 Average Engagement by Platform

This chart and its supporting table show:

- average likes
- average retweets

Key insight:

- the project includes simple descriptive statistics, not only NLP

### 12.9 Top Positive and Top Neutral Keywords

These charts show keyword profiles by sentiment label.

Key insight:

- the sentiment dataset is generic social-media data rather than a domain-specific startup feedback corpus

### 12.10 Current Session Retrieval Mix

This chart shows which source families were used during the live demo session.

Key insight:

- it visually explains the answer style seen in the chatbot

### 12.11 Key Insights and Diagnostic Notes

This section converts the charts into concise textual takeaways.

This is useful in presentation because it helps move from visualization to interpretation.

---

## 13. Expected Business Insights

The project’s value is not only technical.  
It also produces entrepreneurship-related insights.

### 13.1 Startup-pattern insights

From YC data, users can learn:

- what kinds of startups exist
- what themes are common
- what examples may be useful for benchmarking

### 13.2 Official procedural insights

From Hong Kong official sources, users can learn:

- how to register a company
- what filing and compliance steps matter
- how annual returns and e-services work

### 13.3 Practical launch insights

From business-guide sources, users can learn:

- how to start a business
- how to plan retail launch
- how to think about pricing and competition
- how to define target customers
- how to manage inventory and supply chain

### 13.4 Feedback and sentiment insights

From the sentiment layer, users can learn:

- how feedback-style questions can be detected
- how platform-level sentiment data can be described
- how keyword analysis and engagement analysis can support dashboard interpretation

### 13.5 Conversation-flow insights

From follow-up prediction, users can learn what logical next questions to ask.

This makes the system feel more like an entrepreneurship advisor and less like a static QA tool.

---

## 14. Demo Workflow

The project is designed to be demonstrable in a classroom setting.

### Suggested demo sequence

1. Ask an official Hong Kong question  
   Example: `How do I register a company in Hong Kong?`

2. Ask a business-guide question  
   Example: `How do I open a retail store and plan pricing for target customers?`

3. Ask a sentiment-oriented question  
   Example: `What are customers complaining about most in online retail reviews?`

4. Ask a YC pattern question  
   Example: `How can I start a healthcare startup in Hong Kong?`

This sequence demonstrates:

- official-source routing
- business-guide routing
- sentiment triggering
- startup-pattern retrieval
- analytics dashboard updates

---

## 15. How to Run the Project

### Installation

Install dependencies:

```cmd
python -m pip install -r requirements.txt
python -m playwright install chromium
```

### Build and run sequence

Recommended full sequence:

```cmd
python build_keywords.py
python external_sources\build_hk_official_kb.py
python external_sources\build_business_guide_kb.py
python sentiment_assets\build_sentiment_assets.py
python classification\classify_inquiry_topics.py
python rag\build_retrieval_index.py
python -m streamlit run app.py
```

### Faster demo sequence

If outputs are already built:

```cmd
python rag\build_retrieval_index.py
python -m streamlit run app.py
```

### Optional local model polishing

If local answer polishing is needed, the project supports Ollama with `qwen2.5:3b`.

This part is optional.  
The chatbot still works without local model polishing because the default answer pipeline is deterministic and evidence-grounded.

---

## 16. Current Strengths of the Project

The current project already has several strong points:

- multi-source knowledge design
- a complete end-to-end pipeline
- clear objective mapping to Project 8
- classification + retrieval + answer + follow-up integration
- a working chatbot front end
- an analytics dashboard
- sentiment analysis as a supplementary module
- evidence gap logic
- explainable source-family distinctions

These strengths make the system suitable for demonstration and course submission.

---

## 17. Current Limitations

Although the project is functional and explainable, there are still some limitations.

### 17.1 Source imbalance

YC is much larger than the Hong Kong official and business-guide corpora.  
This means the system must carefully route smaller high-value sources.

### 17.2 Classifier weakness on some labels

Some topic labels still have weak F1 scores.

### 17.3 Sentiment dataset imbalance

The sentiment dataset is heavily neutral-heavy and not domain-specific enough.

### 17.4 External page access restrictions

Some business-guide webpages may block direct scraping.

### 17.5 Retrieval evidence limitations

Some user questions, especially complaint-ranking questions, would ideally require a dedicated review corpus that the current system does not yet include.

---

## 18. Future Improvement Directions

If this project were extended further, the most valuable improvements would be:

- add more Hong Kong local sources
- expand the business-guide source family
- add a domain-specific retail review or startup review corpus
- improve weak-topic classification through better labels or more examples
- add more rigorous retrieval evaluation benchmarks
- enhance dashboard insight narration

These improvements would make the system stronger both academically and practically.

---

## 19. Conclusion

This project is a **multi-source entrepreneurship chatbot and analytics system** built for ARIN 7102 Project 8.

It combines:

- data collection
- data cleansing
- keyword extraction
- topic classification
- multi-source retrieval
- evidence-grounded answer generation
- follow-up prediction
- sentiment analysis
- and dashboard visualization

The final system is suitable for course submission because it is:

- runnable
- explainable
- demonstrable
- and clearly aligned with the six official project objectives

Most importantly, the project does not simply answer questions from one source.  
It demonstrates how entrepreneurship support can be improved by combining:

- startup patterns,
- official local procedures,
- practical business guidance,
- and supplementary sentiment analytics

into one integrated chatbot experience.
