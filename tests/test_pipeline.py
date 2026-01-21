"""
Integration tests for the document processing pipeline.

Tests the full flow: classify → extract → validate.
Uses mock extractor by default to avoid API costs.
"""

import pytest
from src.schemas import DocumentType, ProcessingResult, ValidationResult
from src.pipeline import DocumentPipeline, Document
from src.classifier import DocumentClassifier
from src.extractor import MockExtractor


class TestDocumentPipeline:
    """Tests for the DocumentPipeline class."""
    
    @pytest.fixture
    def pipeline_with_mock(self):
        """Create a pipeline with mock extractor and trained classifier."""
        # Train classifier on minimal data
        classifier = DocumentClassifier()
        texts = [
            "INVOICE Total: $500", "Invoice #123 Amount Due: $100",
            "SHIPPING Tracking: ABC", "Ship To: NYC Carrier: UPS",
            "INVENTORY Stock: 100", "Warehouse Report Units: 500",
            "PURCHASE ORDER PO#123", "PO Vendor: Acme Total: $1000",
        ]
        labels = [
            "invoice", "invoice",
            "shipping_order", "shipping_order", 
            "inventory_report", "inventory_report",
            "purchase_order", "purchase_order",
        ]
        classifier.train(texts, labels, test_size=0.25)
        
        # Create pipeline with mock extractor
        pipeline = DocumentPipeline(
            classifier=classifier,
            extractor=MockExtractor(),
        )
        
        return pipeline
    
    def test_classify_only(self, pipeline_with_mock):
        """Test classification without extraction."""
        text = "INVOICE Invoice #: INV-999 Total: $1000"
        result = pipeline_with_mock.classify(text)
        
        assert result.document_type == DocumentType.INVOICE
        assert result.confidence > 0
    
    def test_process_with_extraction(self, pipeline_with_mock):
        """Test full processing with extraction."""
        text = "INVOICE Invoice #: INV-999 Total: $1000"
        result = pipeline_with_mock.process(text)
        
        assert isinstance(result, ProcessingResult)
        assert result.classification.document_type == DocumentType.INVOICE
        assert result.extracted_data is not None
        assert result.validation.is_valid
    
    def test_process_skip_extraction(self, pipeline_with_mock):
        """Test processing with extraction skipped."""
        text = "INVOICE Total: $500"
        result = pipeline_with_mock.process(text, skip_extraction=True)
        
        assert result.classification.document_type == DocumentType.INVOICE
        assert result.extracted_data == {}
        assert "Extraction skipped" in result.validation.errors
    
    def test_process_batch(self, pipeline_with_mock):
        """Test batch processing."""
        texts = [
            "INVOICE Total: $500",
            "SHIPPING Tracking: 123",
            "INVENTORY Stock: 100",
        ]
        
        results = pipeline_with_mock.process_batch(texts, skip_extraction=True)
        
        assert len(results) == 3
        assert all(isinstance(r, ProcessingResult) for r in results)
    
    def test_pipeline_without_classifier_raises(self):
        """Test that processing without classifier raises error."""
        pipeline = DocumentPipeline(use_mock_extractor=True)
        
        with pytest.raises(ValueError, match="Classifier not loaded"):
            pipeline.classify("Some text")
    
    def test_extract_directly(self, pipeline_with_mock):
        """Test direct extraction (bypassing classification)."""
        text = "PO #123 Vendor: Test Total: $500"
        result = pipeline_with_mock.extract(text, DocumentType.PURCHASE_ORDER)
        
        assert "data" in result
        assert "validation" in result


class TestDocument:
    """Tests for the Document dataclass."""
    
    def test_document_creation(self):
        """Test creating a Document."""
        doc = Document(
            text="INVOICE #123",
            id="doc_1",
            original_label="invoice"
        )
        
        assert doc.text == "INVOICE #123"
        assert doc.id == "doc_1"
        assert doc.original_label == "invoice"
    
    def test_document_defaults(self):
        """Test Document with default values."""
        doc = Document(text="Some text")
        
        assert doc.text == "Some text"
        assert doc.id == ""
        assert doc.original_label is None


class TestProcessingResult:
    """Tests for ProcessingResult schema."""
    
    def test_full_processing_result(self):
        """Test creating a complete ProcessingResult."""
        from src.schemas import ClassificationResult
        
        result = ProcessingResult(
            classification=ClassificationResult(
                document_type=DocumentType.INVOICE,
                confidence=0.95
            ),
            extracted_data={"invoice_number": "INV-001"},
            validation=ValidationResult(is_valid=True, errors=[]),
            raw_text="INVOICE #INV-001"
        )
        
        assert result.classification.document_type == DocumentType.INVOICE
        assert result.extracted_data["invoice_number"] == "INV-001"
        assert result.validation.is_valid
        assert result.raw_text == "INVOICE #INV-001"


# Mark integration tests that require real API calls
@pytest.mark.integration
class TestIntegration:
    """Integration tests that require real API calls.
    
    Run with: pytest -m integration
    Requires OPENAI_API_KEY environment variable.
    """
    
    @pytest.fixture
    def real_pipeline(self):
        """Create a pipeline with real extractor."""
        import os
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        classifier = DocumentClassifier()
        texts = [
            "INVOICE Total: $500", "Invoice #123",
            "SHIPPING Tracking: ABC", "Ship To: NYC",
            "INVENTORY Stock: 100", "Warehouse Report",
            "PURCHASE ORDER PO#123", "PO Vendor: Acme",
        ]
        labels = [
            "invoice", "invoice",
            "shipping_order", "shipping_order",
            "inventory_report", "inventory_report",
            "purchase_order", "purchase_order",
        ]
        classifier.train(texts, labels, test_size=0.25)
        
        return DocumentPipeline(classifier=classifier)
    
    def test_real_extraction(self, real_pipeline):
        """Test with real OpenAI API call."""
        text = """
        INVOICE
        
        Invoice Number: INV-2024-001
        Date: January 15, 2024
        
        Bill To: Acme Corporation
        
        Description          Qty    Price    Total
        Widget A             10     $25.00   $250.00
        Widget B             5      $50.00   $250.00
        
        TOTAL: $500.00
        
        Payment Due: February 15, 2024
        """
        
        result = real_pipeline.process(text)
        
        assert result.classification.document_type == DocumentType.INVOICE
        assert result.validation.is_valid
        assert "invoice_number" in result.extracted_data


