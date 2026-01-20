"""
DocStruct - Document Processing Pipeline
Clean, functional UI for document classification and extraction.
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path

from src.schemas import DocumentType
from src.pipeline import DocumentPipeline, load_sample_documents, Document

# Page config
st.set_page_config(
    page_title="DocStruct",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Clean CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    .stApp {
        background: #0f1117;
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        max-width: 1400px;
        padding: 1rem 2rem;
    }
    
    /* Typography */
    h1, h2, h3 { color: #fafafa !important; font-family: 'Inter', sans-serif !important; }
    p, span, label { color: #a1a1aa !important; }
    
    /* Header */
    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 0 1rem 0;
        border-bottom: 1px solid #27272a;
        margin-bottom: 1.5rem;
    }
    .app-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #fafafa;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .app-subtitle {
        font-size: 0.85rem;
        color: #71717a;
    }
    
    /* Stats row */
    .stats-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .stat-chip {
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 8px;
        padding: 0.75rem 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .stat-chip-icon {
        font-size: 1.25rem;
    }
    .stat-chip-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #fafafa;
    }
    .stat-chip-label {
        font-size: 0.75rem;
        color: #71717a;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Type badges */
    .type-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .type-invoice { background: #1e3a5f; color: #60a5fa; }
    .type-shipping { background: #14532d; color: #4ade80; }
    .type-purchase { background: #3b0764; color: #c084fc; }
    .type-inventory { background: #422006; color: #fbbf24; }
    
    /* Buttons */
    .stButton > button {
        background: #3b82f6;
        color: white !important;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.15s;
    }
    .stButton > button:hover {
        background: #2563eb;
    }
    
    /* Secondary button style */
    .secondary-btn button {
        background: #27272a !important;
        border: 1px solid #3f3f46 !important;
    }
    
    /* Table styling */
    .stDataFrame { border-radius: 8px; }
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 8px;
    }
    
    /* Text area */
    .stTextArea textarea {
        background: #18181b !important;
        border: 1px solid #27272a !important;
        color: #e4e4e7 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
    }
    
    /* Select box */
    .stSelectbox > div > div {
        background: #18181b;
        border: 1px solid #27272a;
    }
    
    /* JSON display */
    .stJson {
        background: #18181b !important;
        border: 1px solid #27272a;
        border-radius: 8px;
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background: #18181b !important;
        border: 1px solid #27272a !important;
        color: #a1a1aa !important;
    }
    .stDownloadButton > button:hover {
        border-color: #3b82f6 !important;
        color: #3b82f6 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #18181b !important;
        border: 1px solid #27272a !important;
        border-radius: 8px !important;
    }
    
    /* Alerts */
    [data-testid="stAlert"] { border-radius: 6px; }
    
    /* Hide streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }
    
    /* Two column layout */
    .doc-panel {
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 8px;
        padding: 1rem;
        height: 100%;
    }
    
    /* Confidence bar */
    .confidence-bar {
        height: 4px;
        background: #27272a;
        border-radius: 2px;
        overflow: hidden;
        margin-top: 0.25rem;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)


# Document type config
TYPE_CONFIG = {
    DocumentType.INVOICE: {"icon": "📊", "label": "Invoice", "class": "type-invoice"},
    DocumentType.SHIPPING_ORDER: {"icon": "📦", "label": "Shipping", "class": "type-shipping"},
    DocumentType.PURCHASE_ORDER: {"icon": "🛒", "label": "Purchase", "class": "type-purchase"},
    DocumentType.INVENTORY_REPORT: {"icon": "📋", "label": "Inventory", "class": "type-inventory"},
}


@st.cache_resource
def get_pipeline():
    return DocumentPipeline(use_mock_extractor=False)


def load_and_classify(n_samples: int):
    """Load and classify documents."""
    pipeline = get_pipeline()
    model_path = Path("models/classifier.pkl")
    
    if not pipeline.classifier:
        pipeline.load_classifier(model_path=model_path, train_sample_size=500)
    
    docs = load_sample_documents(n_samples)
    texts = [doc.text for doc in docs]
    classifications = pipeline.classify_batch(texts)
    
    return docs, classifications


def main():
    # =========================================================================
    # Header
    # =========================================================================
    st.markdown("""
    <div class="app-header">
        <div>
            <div class="app-title">📄 DocStruct</div>
            <div class="app-subtitle">Document Classification & Extraction Pipeline</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # =========================================================================
    # Controls Row
    # =========================================================================
    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 5])
    
    with col1:
        n_samples = st.selectbox(
            "Sample size",
            options=[10, 20, 30, 50],
            index=1,
            help="Number of documents to load from HuggingFace dataset"
        )
    
    with col2:
        load_btn = st.button("📥 Load Data", use_container_width=True)
    
    with col3:
        if st.session_state.get("docs"):
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.docs = None
                st.session_state.classifications = None
                st.session_state.selected_idx = None
                st.session_state.extraction = None
                st.rerun()
    
    # =========================================================================
    # Load Data
    # =========================================================================
    if load_btn:
        with st.spinner("Loading documents & training classifier..."):
            docs, classifications = load_and_classify(n_samples)
            st.session_state.docs = docs
            st.session_state.classifications = classifications
            st.session_state.selected_idx = 0
            st.session_state.extraction = None
    
    # =========================================================================
    # Main Content
    # =========================================================================
    if st.session_state.get("docs"):
        docs = st.session_state.docs
        classifications = st.session_state.classifications
        
        # Stats
        type_counts = {}
        for c in classifications:
            t = TYPE_CONFIG.get(c.document_type, {}).get("label", "Other")
            type_counts[t] = type_counts.get(t, 0) + 1
        
        avg_conf = sum(c.confidence for c in classifications) / len(classifications)
        
        # Stats chips - build list first, then join
        type_chips = []
        for doc_type, count in type_counts.items():
            icon = {"Invoice": "📊", "Shipping": "📦", "Purchase": "🛒", "Inventory": "📋"}.get(doc_type, "📄")
            type_chips.append(f'<div class="stat-chip"><span class="stat-chip-icon">{icon}</span><div><div class="stat-chip-value">{count}</div><div class="stat-chip-label">{doc_type}</div></div></div>')
        
        stats_html = f'''<div class="stats-row"><div class="stat-chip"><span class="stat-chip-icon">📄</span><div><div class="stat-chip-value">{len(docs)}</div><div class="stat-chip-label">Documents</div></div></div><div class="stat-chip"><span class="stat-chip-icon">🎯</span><div><div class="stat-chip-value">{avg_conf:.0%}</div><div class="stat-chip-label">Avg Confidence</div></div></div>{"".join(type_chips)}</div>'''
        st.markdown(stats_html, unsafe_allow_html=True)
        
        # Two-column layout: Table + Detail
        left_col, right_col = st.columns([1.2, 1])
        
        # ---------------------------------------------------------------------
        # Left: Document Table
        # ---------------------------------------------------------------------
        with left_col:
            st.markdown("#### 📋 Documents")
            
            # Filter
            filter_type = st.selectbox(
                "Filter",
                ["All Types"] + list(type_counts.keys()),
                label_visibility="collapsed"
            )
            
            # Build table data
            table_data = []
            for i, (doc, clf) in enumerate(zip(docs, classifications)):
                config = TYPE_CONFIG.get(clf.document_type, {"label": "Other", "icon": "📄"})
                
                # Skip if filtering
                if filter_type != "All Types" and config["label"] != filter_type:
                    continue
                
                table_data.append({
                    "idx": i,
                    "ID": f"DOC-{i+1:03d}",
                    "Type": f"{config['icon']} {config['label']}",
                    "Confidence": f"{clf.confidence:.0%}",
                    "Preview": doc.text[:60].replace("\n", " ") + "...",
                })
            
            # Create DataFrame
            if table_data:
                df = pd.DataFrame(table_data)
                
                # Document selector
                selected_id = st.selectbox(
                    "Select document to view details →",
                    options=[row["ID"] for row in table_data],
                    index=0,
                )
                
                # Update selected index
                selected_row = next((r for r in table_data if r["ID"] == selected_id), None)
                if selected_row:
                    st.session_state.selected_idx = selected_row["idx"]
                
                # Show table
                st.dataframe(
                    df[["ID", "Type", "Confidence", "Preview"]],
                    use_container_width=True,
                    hide_index=True,
                    height=350,
                )
                
                # Export buttons
                st.markdown("##### Export")
                exp_col1, exp_col2 = st.columns(2)
                
                with exp_col1:
                    # CSV export
                    csv_data = []
                    for i, (doc, clf) in enumerate(zip(docs, classifications)):
                        config = TYPE_CONFIG.get(clf.document_type, {"label": "Other"})
                        csv_data.append({
                            "ID": f"DOC-{i+1:03d}",
                            "Type": config["label"],
                            "Confidence": f"{clf.confidence:.4f}",
                            "Text": doc.text,
                        })
                    csv_df = pd.DataFrame(csv_data)
                    st.download_button(
                        "📊 CSV",
                        csv_df.to_csv(index=False),
                        "docstruct_export.csv",
                        "text/csv",
                        use_container_width=True,
                    )
                
                with exp_col2:
                    # JSON export
                    json_data = []
                    for i, (doc, clf) in enumerate(zip(docs, classifications)):
                        json_data.append({
                            "id": f"DOC-{i+1:03d}",
                            "type": clf.document_type.value,
                            "confidence": round(clf.confidence, 4),
                            "text": doc.text,
                        })
                    st.download_button(
                        "📦 JSON",
                        json.dumps(json_data, indent=2),
                        "docstruct_export.json",
                        "application/json",
                        use_container_width=True,
                    )
        
        # ---------------------------------------------------------------------
        # Right: Document Detail + Extraction
        # ---------------------------------------------------------------------
        with right_col:
            idx = st.session_state.get("selected_idx", 0)
            if idx is not None and idx < len(docs):
                doc = docs[idx]
                clf = classifications[idx]
                config = TYPE_CONFIG.get(clf.document_type, {"label": "Other", "icon": "📄", "class": "type-invoice"})
                
                st.markdown("#### 🔍 Document Detail")
                
                # Type badge and confidence
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                    <span class="type-badge {config['class']}">{config['icon']} {config['label']}</span>
                    <span style="color: #71717a; font-size: 0.85rem;">
                        Confidence: <strong style="color: #fafafa;">{clf.confidence:.1%}</strong>
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence bar
                st.markdown(f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {clf.confidence*100}%;"></div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Raw text
                with st.expander("📝 Raw Document Text", expanded=False):
                    st.text_area(
                        "raw",
                        doc.text,
                        height=200,
                        disabled=True,
                        label_visibility="collapsed",
                    )
                
                # Extraction section
                st.markdown("##### 🧠 LLM Extraction")
                
                if st.button("🚀 Extract Structured Data", use_container_width=True):
                    import os
                    if not os.getenv("OPENAI_API_KEY"):
                        st.error("⚠️ Set OPENAI_API_KEY in .env file")
                    else:
                        with st.spinner("Extracting with GPT-4o-mini..."):
                            try:
                                pipeline = get_pipeline()
                                result = pipeline.process(doc.text)
                                st.session_state.extraction = {
                                    "idx": idx,
                                    "result": result,
                                }
                            except Exception as e:
                                st.error(f"Error: {e}")
                
                # Show extraction result (only if it matches current document)
                extraction = st.session_state.get("extraction")
                if extraction and extraction.get("idx") == idx:
                    result = extraction["result"]
                    if result.validation.is_valid:
                        st.success("✅ Extraction successful")
                    else:
                        st.warning("⚠️ Partial extraction")
                    
                    st.json(result.extracted_data)
                    
                    # Download extracted JSON
                    st.download_button(
                        "📥 Download Extracted JSON",
                        json.dumps(result.extracted_data, indent=2),
                        f"doc_{idx+1}_extracted.json",
                        "application/json",
                        use_container_width=True,
                    )
    
    else:
        # Empty state
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; color: #71717a;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📁</div>
            <div style="font-size: 1.1rem; margin-bottom: 0.5rem;">No documents loaded</div>
            <div style="font-size: 0.85rem;">Select a sample size and click <strong>Load Data</strong> to get started</div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
