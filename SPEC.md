# Document Processing Pipeline - Technical Specification

## Overview

A hybrid ML + LLM document processing pipeline that classifies business documents into 4 types and extracts structured metadata for ingestion into Collibra's data catalog.

**One-sentence summary**: Messy text goes in → ML figures out what type of document it is → LLM extracts the important fields → Schema validates the output → Clean, structured JSON comes out.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DOCUMENT PROCESSING PIPELINE                  │
└─────────────────────────────────────────────────────────────────────┘

   INPUT                    PROCESS                         OUTPUT
   ─────                    ───────                         ──────

┌──────────────┐      ┌─────────────────┐
│ Raw Document │ ───► │ 1. CLASSIFIER   │
│ (text blob)  │      │    (ML: TF-IDF) │
└──────────────┘      └────────┬────────┘
                               │
                               ▼
                      "This is an INVOICE"
                      (confidence: 94%)
                               │
                               ▼
                      ┌─────────────────┐
                      │ 2. EXTRACTOR    │
                      │    (LLM: GPT)   │
                      └────────┬────────┘
                               │
                      Prompt: "Extract invoice_number,
                               date, vendor, total..."
                               │
                               ▼
                      ┌─────────────────┐
                      │ 3. VALIDATOR    │
                      │   (Schema check)│
                      └────────┬────────┘
                               │
                               ▼
                      ┌─────────────────────────┐
                      │  STRUCTURED JSON OUTPUT │
                      └─────────────────────────┘
```

---

## Data Source

**Dataset**: [HuggingFace - AyoubChLin/CompanyDocuments](https://huggingface.co/datasets/AyoubChLin/CompanyDocuments)

### Dataset Structure

| Column | Type | Description |
|--------|------|-------------|
| `file_content` | string | Raw document content (extracted text from business documents) |
| `document_type` | string | Document category (`Invoices`, `Shipping Orders`, `Purchase Orders`) |
| `file_name` | string | Original filename (e.g., `order_10562.pdf`) |
| `extracted_data` | string | Pre-extracted JSON data (for reference) |

### Example Dataset Row

```python
{
    "file_content": "Order ID: 10562\nShipping Details:\nShip Name: Reggiani Caseifici\nShip Address: Strada Provinciale 124\n...",
    "document_type": "Shipping Orders",
    "file_name": "order_10562.pdf"
}
```

### Loading the Dataset

```python
from datasets import load_dataset

dataset = load_dataset("AyoubChLin/CompanyDocuments")
train_data = dataset["train"]

# Access a sample
sample = train_data[0]
text = sample["file_content"]
label = sample["document_type"]
```

---

## Document Types

The pipeline classifies documents into exactly **4 types**:

| Type | Description |
|------|-------------|
| `shipping_order` | Documents related to shipping, tracking, and delivery |
| `invoice` | Bills requesting payment for goods/services |
| `inventory_report` | Reports on stock levels and warehouse inventory |
| `purchase_order` | Orders placed to vendors for goods/services |

---

## Technical Components

### 1. Classifier (ML-based)

**Technology**: TF-IDF + Logistic Regression (scikit-learn)

#### What is TF-IDF?

**TF-IDF** = **T**erm **F**requency × **I**nverse **D**ocument **F**requency

A method to convert text into numerical vectors that ML models can understand.

| Component | What it measures | Example |
|-----------|-----------------|---------|
| **TF** (Term Frequency) | How often a word appears in *this* document | "invoice" appears 5 times → high TF |
| **IDF** (Inverse Document Frequency) | How rare a word is across *all* documents | "the" appears everywhere → low IDF; "invoice" is rarer → high IDF |
| **TF × IDF** | Words that are frequent here but rare elsewhere | High score = distinctive word for this doc type |

**Why TF-IDF + Logistic Regression?**
- Fast inference (milliseconds)
- No API calls needed for classification
- Works offline
- Easy to retrain with new data
- Interpretable (can see which words drive decisions)

---

### 2. Extractor (LLM-based)

**Technology**: OpenAI GPT-4o-mini

**Why OpenAI GPT-4o-mini?**
- Native JSON mode with structured outputs
- Fast response times (~1-2 seconds)
- Cost-effective: $0.15 per 1M input tokens
- Excellent at following extraction schemas
- Pydantic model support for type-safe outputs

**Extraction Method**: 
- Pass document text + Pydantic schema to OpenAI
- LLM returns structured JSON matching the schema
- Built-in validation via Pydantic

---

### 3. Schemas (Pydantic Models)

Schemas define the expected structure of extracted data. Each document type has its own schema.

#### Invoice Schema
```python
class Invoice:
    invoice_number: str      # Required - e.g., "INV-2024-001"
    date: str                # Required - e.g., "2024-01-15"
    vendor: str              # Required - e.g., "Acme Corp"
    total_amount: float      # Required - e.g., 1234.56
    line_items: list[LineItem]  # Optional - items purchased
    due_date: str | None     # Optional - payment due date
```

#### Shipping Order Schema
```python
class ShippingOrder:
    tracking_number: str     # Required - e.g., "1Z999AA10123456784"
    ship_date: str           # Required - e.g., "2024-01-15"
    origin: str              # Required - origin address/location
    destination: str         # Required - delivery address
    carrier: str             # Required - e.g., "UPS", "FedEx"
    weight: float | None     # Optional - package weight
    status: str | None       # Optional - e.g., "In Transit"
```

#### Inventory Report Schema
```python
class InventoryReport:
    report_date: str         # Required - e.g., "2024-01-15"
    warehouse: str           # Required - warehouse location/ID
    items: list[InventoryItem]  # Required - list of inventory items
    total_units: int         # Required - total count
    total_value: float | None   # Optional - total inventory value
```

#### Purchase Order Schema
```python
class PurchaseOrder:
    po_number: str           # Required - e.g., "PO-2024-001"
    date: str                # Required - order date
    vendor: str              # Required - supplier name
    items: list[LineItem]    # Required - items ordered
    total_value: float       # Required - total order value
    ship_to: str | None      # Optional - delivery address
```

---

## Input/Output Examples

### Example Input Document

```
INVOICE

Invoice #: INV-2024-0847
Date: January 15, 2024
Bill To: TechStart Inc.

Description              Qty    Price     Total
-------------------------------------------------
Cloud Services           1      $500.00   $500.00
Support Package          1      $150.00   $150.00
-------------------------------------------------
                         TOTAL: $650.00

Payment Due: February 15, 2024
```

### Example Output

```json
{
  "classification": {
    "document_type": "invoice",
    "confidence": 0.94
  },
  "extracted_data": {
    "invoice_number": "INV-2024-0847",
    "date": "2024-01-15",
    "vendor": "TechStart Inc.",
    "total_amount": 650.00,
    "due_date": "2024-02-15",
    "line_items": [
      {"description": "Cloud Services", "quantity": 1, "price": 500.00},
      {"description": "Support Package", "quantity": 1, "price": 150.00}
    ]
  },
  "validation": {
    "is_valid": true,
    "errors": []
  }
}
```

---

## UI Design (Streamlit)

The UI follows a two-step flow: Load & Classify → Select & Extract

```
┌─────────────────────────────────────────────────────────────┐
│  📄 Document Processor                                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: Load & Classify                                    │
│  [🔄 Load Sample Documents]                                 │
│                                                             │
│  ✅ Loaded and classified 20 documents                      │
│     📊 Invoice: 5 | 📦 Shipping: 6 | 🛒 PO: 4 | 📋 Inv: 5  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 2: View & Extract                                     │
│                                                             │
│  Select Document: [ Invoice #3 - INV-2024-0847...    ▼]    │
│                   ┌──────────────────────────────────┐      │
│                   │ Invoice #1 - INV-2024-0123...    │      │
│                   │ Invoice #2 - INV-2024-0456...    │      │
│                   │ Shipping #1 - TRK-9876543...     │      │
│                   │ ...                              │      │
│                   └──────────────────────────────────┘      │
│                                                             │
│  ─────────── Document ───────────                           │
│  Type: 📊 INVOICE (94% confidence)                          │
│                                                             │
│  Raw Text:                                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ INVOICE                                               │  │
│  │ Invoice #: INV-2024-0847                             │  │
│  │ ...                                                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ─────────── Extracted Data ───────────                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ {                                                     │  │
│  │   "invoice_number": "INV-2024-0847",                 │  │
│  │   "date": "2024-01-15",                              │  │
│  │   ...                                                │  │
│  │ }                                                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  Validation: ✅ Schema Valid                                │
└─────────────────────────────────────────────────────────────┘
```

### UI Flow

1. **Load**: Click button → Fetches samples from HuggingFace dataset
2. **Classify**: ML classifier categorizes all documents (fast, batch)
3. **Dropdown**: Shows all docs labeled by type (e.g., "Invoice #3 - INV-2024...")
4. **Select**: Choose a document → Shows raw text
5. **Extract**: LLM extracts structured data → Shows JSON output
6. **Validate**: Pydantic validates schema → Shows pass/fail

---

## Testing Strategy

### Test Coverage

| Test Level | Component | What We Test | Method |
|------------|-----------|--------------|--------|
| **Unit** | Classifier | Given known text → returns correct label | pytest, no mocking |
| **Unit** | Extractor | Given text + type → returns valid schema | pytest, mock OpenAI |
| **Unit** | Schemas | Pydantic models validate/reject correctly | pytest |
| **Integration** | Pipeline | End-to-end: text → classification → extraction → JSON | Real LLM call |

### Example Test Cases

```python
# test_classifier.py
def test_classify_invoice():
    text = "INVOICE\nInvoice #: INV-123\nTotal: $500"
    result = classifier.classify(text)
    assert result.document_type == "invoice"
    assert result.confidence > 0.8

def test_classify_shipping_order():
    text = "SHIPPING ORDER\nTracking: 1Z999\nCarrier: UPS"
    result = classifier.classify(text)
    assert result.document_type == "shipping_order"

# test_extractor.py
def test_extract_invoice_fields(mock_openai):
    text = "Invoice #: INV-123\nDate: 2024-01-15\nTotal: $500"
    result = extractor.extract(text, "invoice")
    assert result.invoice_number == "INV-123"
    assert result.total_amount == 500.0

# test_pipeline.py (integration)
def test_full_pipeline():
    text = "INVOICE\nInvoice #: INV-123\nTotal: $500"
    result = pipeline.process(text)
    assert result.classification.document_type == "invoice"
    assert result.extracted_data.invoice_number == "INV-123"
    assert result.validation.is_valid == True
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only unit tests (no API calls)
pytest tests/ -v -m "not integration"

# Run integration tests (requires OPENAI_API_KEY)
pytest tests/ -v -m "integration"
```

---

## Project Structure

```
docstruct/
├── src/
│   ├── __init__.py
│   ├── classifier.py      # TF-IDF + LogReg classifier
│   ├── extractor.py       # OpenAI LLM extraction
│   ├── schemas.py         # Pydantic models for 4 doc types
│   └── pipeline.py        # Orchestrates: classify → extract → validate
├── app.py                 # Streamlit UI
├── tests/
│   ├── __init__.py
│   ├── test_classifier.py # Classifier unit tests
│   ├── test_extractor.py  # Extractor unit tests (mocked)
│   └── test_pipeline.py   # Integration tests
├── requirements.txt       # Dependencies
├── .env.example           # API key template
├── ASSESSMENT.md          # Original problem statement
├── SPEC.md                # This document
└── README.md              # Setup & usage instructions
```

---

## Dependencies

```
openai>=1.0.0          # LLM extraction
scikit-learn>=1.3.0    # TF-IDF + classifier
pydantic>=2.0.0        # Schema validation
streamlit>=1.28.0      # Web UI
datasets>=2.14.0       # HuggingFace data loading
python-dotenv>=1.0.0   # Environment variables
pytest>=7.4.0          # Testing
```

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Classification accuracy | ≥ 90% on test set |
| Extraction precision | ≥ 85% for required fields |
| Schema validation | 100% valid JSON output |
| Latency per document | < 3 seconds |
| Error rate | < 5% pipeline failures |

---

## Implementation Timeline (60 minutes)

| Priority | Component | Time |
|----------|-----------|------|
| P0 | Schemas + Pipeline skeleton | 10 min |
| P0 | Classifier (TF-IDF + LogReg) | 15 min |
| P0 | LLM Extractor with structured output | 15 min |
| P1 | Streamlit UI | 10 min |
| P1 | Basic tests | 5 min |
| P2 | README + polish | 5 min |

