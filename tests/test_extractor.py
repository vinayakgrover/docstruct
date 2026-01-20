"""
Unit tests for the document extractor.

Uses MockExtractor to test extraction logic without API calls.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.schemas import (
    DocumentType,
    Invoice,
    ShippingOrder,
    InventoryReport,
    PurchaseOrder,
    ValidationResult,
)
from src.extractor import DocumentExtractor, MockExtractor


class TestMockExtractor:
    """Tests using the mock extractor (no API calls)."""
    
    @pytest.fixture
    def extractor(self):
        """Create a mock extractor."""
        return MockExtractor()
    
    def test_extract_invoice(self, extractor):
        """Test extracting invoice data."""
        text = "INVOICE #123 Total: $1000"
        extracted, validation = extractor.extract(text, DocumentType.INVOICE)
        
        assert isinstance(extracted, Invoice)
        assert validation.is_valid
        assert extracted.invoice_number == "INV-MOCK-001"
    
    def test_extract_shipping_order(self, extractor):
        """Test extracting shipping order data."""
        text = "SHIPPING Tracking: ABC123"
        extracted, validation = extractor.extract(text, DocumentType.SHIPPING_ORDER)
        
        assert isinstance(extracted, ShippingOrder)
        assert validation.is_valid
        assert extracted.tracking_number == "MOCK123456789"
    
    def test_extract_inventory_report(self, extractor):
        """Test extracting inventory report data."""
        text = "INVENTORY REPORT Stock: 500"
        extracted, validation = extractor.extract(text, DocumentType.INVENTORY_REPORT)
        
        assert isinstance(extracted, InventoryReport)
        assert validation.is_valid
    
    def test_extract_purchase_order(self, extractor):
        """Test extracting purchase order data."""
        text = "PO #456 Total: $2000"
        extracted, validation = extractor.extract(text, DocumentType.PURCHASE_ORDER)
        
        assert isinstance(extracted, PurchaseOrder)
        assert validation.is_valid
        assert extracted.po_number == "PO-MOCK-001"
    
    def test_extract_raw_returns_dict(self, extractor):
        """Test that extract_raw returns a dictionary."""
        text = "INVOICE #123"
        result = extractor.extract_raw(text, DocumentType.INVOICE)
        
        assert isinstance(result, dict)
        assert "data" in result
        assert "validation" in result


class TestDocumentExtractor:
    """Tests for the real DocumentExtractor (mocked OpenAI calls)."""
    
    @pytest.fixture
    def mock_openai_response(self):
        """Create a mock OpenAI API response."""
        mock_invoice = Invoice(
            invoice_number="INV-TEST-001",
            date="2024-01-15",
            vendor="Test Vendor",
            total_amount=1500.00,
        )
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed = mock_invoice
        
        return mock_response
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('src.extractor.OpenAI')
    def test_extract_with_mocked_api(self, mock_openai_class, mock_openai_response):
        """Test extraction with mocked OpenAI API."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = mock_openai_response
        mock_openai_class.return_value = mock_client
        
        # Create extractor and extract
        extractor = DocumentExtractor()
        text = "INVOICE #INV-TEST-001 Date: 2024-01-15 Vendor: Test Vendor Total: $1500"
        
        extracted, validation = extractor.extract(text, DocumentType.INVOICE)
        
        # Verify
        assert isinstance(extracted, Invoice)
        assert extracted.invoice_number == "INV-TEST-001"
        assert extracted.total_amount == 1500.00
        assert validation.is_valid
    
    def test_extractor_requires_api_key(self):
        """Test that extractor raises error without API key."""
        with patch.dict('os.environ', {}, clear=True):
            # Remove any existing key
            import os
            if 'OPENAI_API_KEY' in os.environ:
                del os.environ['OPENAI_API_KEY']
            
            with pytest.raises(ValueError, match="API key required"):
                DocumentExtractor()


class TestValidationResult:
    """Tests for ValidationResult schema."""
    
    def test_valid_result(self):
        """Test creating a valid validation result."""
        result = ValidationResult(is_valid=True, errors=[])
        
        assert result.is_valid
        assert result.errors == []
    
    def test_invalid_result_with_errors(self):
        """Test creating an invalid validation result with errors."""
        result = ValidationResult(
            is_valid=False,
            errors=["Missing required field: invoice_number", "Invalid date format"]
        )
        
        assert not result.is_valid
        assert len(result.errors) == 2

