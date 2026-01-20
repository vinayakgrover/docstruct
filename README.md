# DocStruct

**Document Classification & Extraction Pipeline**

Transform unstructured business documents into structured, searchable JSON data.

## Features

- **ML Classification** — TF-IDF + Logistic Regression classifies documents into 4 types
- **LLM Extraction** — GPT-4o-mini extracts structured fields from raw text
- **Export** — Download results as CSV or JSON
- **Streamlit UI** — Interactive web interface

## Document Types

| Type | Example Fields |
|------|---------------|
| 📊 Invoice | order_id, customer, line_items, total |
| 📦 Shipping Order | tracking_number, carrier, destination, recipient |
| 🛒 Purchase Order | po_number, vendor, items, total_value |
| 📋 Inventory Report | warehouse, items, quantities |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
echo "OPENAI_API_KEY=sk-your-key" > .env

# Run the app
streamlit run app.py
```

Open http://localhost:8501

## How It Works

```
Raw Document → [ML Classifier] → Document Type → [LLM Extractor] → Structured JSON
```

1. **Load** sample documents from HuggingFace dataset
2. **Classify** using TF-IDF vectorization + Logistic Regression (87% avg confidence)
3. **Extract** structured data using GPT-4o-mini with Pydantic schemas
4. **Export** to CSV/JSON for downstream use

## Project Structure

```
├── app.py              # Streamlit UI
├── src/
│   ├── classifier.py   # TF-IDF + Logistic Regression
│   ├── extractor.py    # OpenAI LLM extraction
│   ├── pipeline.py     # Orchestration
│   └── schemas.py      # Pydantic models
├── tests/              # Unit tests
├── models/             # Trained classifier
└── requirements.txt
```

## Tech Stack

- **Classification**: scikit-learn (TF-IDF, LogisticRegression)
- **Extraction**: OpenAI GPT-4o-mini
- **Validation**: Pydantic
- **UI**: Streamlit
- **Data**: HuggingFace datasets

## License

MIT
