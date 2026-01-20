"""
Unit tests for the document classifier.

Tests classification logic without requiring the HuggingFace dataset
or trained model - uses synthetic test data.
"""

import pytest
from src.schemas import DocumentType, ClassificationResult
from src.classifier import DocumentClassifier


class TestDocumentClassifier:
    """Tests for DocumentClassifier class."""
    
    @pytest.fixture
    def trained_classifier(self):
        """Create a classifier trained on minimal synthetic data."""
        classifier = DocumentClassifier()
        
        # Minimal training data - just enough to test the pipeline
        texts = [
            # Invoices
            "INVOICE Invoice Number: INV-001 Total Amount: $500.00 Due Date: 2024-01-15",
            "Invoice #12345 Bill To: Customer Inc. Amount Due: $1,200.00",
            "TAX INVOICE Date: 2024-01-01 Subtotal: $100 Tax: $10 Total: $110",
            
            # Shipping Orders  
            "SHIPPING ORDER Tracking Number: 1Z999AA1 Carrier: UPS Destination: NYC",
            "Ship To: 123 Main St Tracking: FEDEX123 Estimated Delivery: Tomorrow",
            "BILL OF LADING Shipper: ABC Corp Consignee: XYZ Inc Weight: 100 lbs",
            
            # Inventory Reports
            "INVENTORY REPORT Warehouse: A1 Date: 2024-01-15 Total Units: 5000",
            "Stock Level Report SKU: ABC123 Quantity on Hand: 250 Reorder Point: 50",
            "Inventory Count Date: 2024-01-01 Location: Warehouse B Items: 1500",
            
            # Purchase Orders
            "PURCHASE ORDER PO Number: PO-2024-001 Vendor: Supplier Inc Total: $2000",
            "Purchase Requisition Buyer: John Doe Items: 10 widgets @ $50 each",
            "PO #98765 Ship To: Our Warehouse Terms: Net 30 Total Value: $5000",
        ]
        
        labels = [
            "invoice", "invoice", "invoice",
            "shipping_order", "shipping_order", "shipping_order",
            "inventory_report", "inventory_report", "inventory_report",
            "purchase_order", "purchase_order", "purchase_order",
        ]
        
        classifier.train(texts, labels, test_size=0.25)
        return classifier
    
    def test_classifier_training(self, trained_classifier):
        """Test that classifier trains successfully."""
        assert trained_classifier.is_trained
        assert trained_classifier.pipeline is not None
    
    def test_classify_invoice(self, trained_classifier):
        """Test classifying an invoice document."""
        text = "INVOICE Invoice #: INV-999 Total: $1000.00 Payment Due: 2024-02-01"
        result = trained_classifier.classify(text)
        
        assert isinstance(result, ClassificationResult)
        assert result.document_type == DocumentType.INVOICE
        assert 0.0 <= result.confidence <= 1.0
    
    def test_classify_shipping_order(self, trained_classifier):
        """Test classifying a shipping order document."""
        text = "SHIPPING ORDER Tracking: 1Z999 Carrier: FedEx Deliver To: 456 Oak St"
        result = trained_classifier.classify(text)
        
        assert isinstance(result, ClassificationResult)
        assert result.document_type == DocumentType.SHIPPING_ORDER
    
    def test_classify_inventory_report(self, trained_classifier):
        """Test classifying an inventory report document."""
        text = "INVENTORY REPORT Date: 2024-01-20 Warehouse: Main Total Stock: 10000 units"
        result = trained_classifier.classify(text)
        
        assert isinstance(result, ClassificationResult)
        assert result.document_type == DocumentType.INVENTORY_REPORT
    
    def test_classify_purchase_order(self, trained_classifier):
        """Test classifying a purchase order document."""
        text = "PURCHASE ORDER PO#: PO-123 Vendor: Acme Corp Order Total: $3500"
        result = trained_classifier.classify(text)
        
        assert isinstance(result, ClassificationResult)
        assert result.document_type == DocumentType.PURCHASE_ORDER
    
    def test_classify_batch(self, trained_classifier):
        """Test batch classification."""
        texts = [
            "INVOICE Total: $500",
            "SHIPPING Tracking: ABC123",
            "INVENTORY Stock: 100 units",
        ]
        
        results = trained_classifier.classify_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, ClassificationResult) for r in results)
    
    def test_untrained_classifier_raises_error(self):
        """Test that classifying with untrained classifier raises error."""
        classifier = DocumentClassifier()
        
        with pytest.raises(ValueError, match="not been trained"):
            classifier.classify("Some document text")
    
    def test_get_top_features(self, trained_classifier):
        """Test getting top features for each class."""
        features = trained_classifier.get_top_features(n=5)
        
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Each class should have features
        for class_label, class_features in features.items():
            assert isinstance(class_features, list)
            assert len(class_features) <= 5


class TestClassificationResult:
    """Tests for ClassificationResult schema."""
    
    def test_valid_classification_result(self):
        """Test creating a valid ClassificationResult."""
        result = ClassificationResult(
            document_type=DocumentType.INVOICE,
            confidence=0.95
        )
        
        assert result.document_type == DocumentType.INVOICE
        assert result.confidence == 0.95
    
    def test_confidence_bounds(self):
        """Test that confidence must be between 0 and 1."""
        # Valid bounds
        ClassificationResult(document_type=DocumentType.INVOICE, confidence=0.0)
        ClassificationResult(document_type=DocumentType.INVOICE, confidence=1.0)
        
        # Invalid bounds should raise
        with pytest.raises(ValueError):
            ClassificationResult(document_type=DocumentType.INVOICE, confidence=1.5)
        
        with pytest.raises(ValueError):
            ClassificationResult(document_type=DocumentType.INVOICE, confidence=-0.1)

