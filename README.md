# AI Pharmacist Assistant

A bilingual, source-grounded medication information demo built with Streamlit, LangChain, FAISS, Hugging Face embeddings, and the DeepSeek API.

The application searches local medicine package inserts before generating an answer. It displays document names, page numbers, and retrieval scores, and fails closed when the available evidence is insufficient.

## Features

- PDF package-insert ingestion with scanned-PDF detection
- Semantic retrieval with configurable distance threshold
- English and Chinese grounded answers
- Page-level source citations and citation validation
- Conservative refusal when evidence is missing
- Environment-based API keys and administrator password
- Offline safety tests and retrieval evaluation
- Small interaction-rule demo, clearly separated from clinical data

## Architecture

```text
Streamlit UI (app.py)
    ├── RAG ingestion and retrieval (rag_core.py)
    ├── grounded generation and validation (llm_service.py)
    ├── demo interaction rules (interaction_rules.py)
    └── environment configuration (config.py)
```

## Run locally

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and add your own values.

3. Start the application:

   ```bash
   streamlit run app.py
   ```

4. Sign in, upload a text-based medicine PDF, select **Add to knowledge base**, and submit a medication-information question.

## Test without spending API credit

```bash
python -m unittest discover -s tests -v
```

The unit tests do not call DeepSeek.

## Evaluate retrieval

Edit `evaluation/golden_questions.example.json` so its expected filenames match your package inserts, then run:

```bash
python evaluation/evaluate_retrieval.py
```

This reports Hit-rate@k and MRR@k without calling an LLM. Use the failures to tune `RAG_MAX_DISTANCE`, chunk size, and document metadata.

## Safety and limitations

This is a portfolio demonstration, not a medical device. It does not diagnose, prescribe, or establish that a medicine or combination is safe. Every output must be checked against the cited document and an authoritative clinical source. The interaction rules are intentionally incomplete demo data. Do not upload real patient information.

