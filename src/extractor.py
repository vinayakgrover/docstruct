"""
Document Extractor using OpenAI LLM with structured outputs.

This module extracts structured data from documents using GPT-4o-mini.
It leverages OpenAI's structured output feature with Pydantic models
to ensure type-safe, validated JSON extraction.

Why OpenAI GPT-4o-mini?
- Native JSON mode with structured outputs
- Fast response times (~1-2 seconds)
- Cost-effective: $0.15 per 1M input tokens
- Excellent at following extraction schemas
- Pydantic model support for type-safe outputs
"""

import os
from typing import Union

from openai import OpenAI
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

from .schemas import (
    DocumentType,
    Invoice,
    ShippingOrder,
    InventoryReport,
    PurchaseOrder,
    ValidationResult,
    DOCUMENT_SCHEMAS,
    get_schema_for_type,
)

# Load environment variables
load_dotenv()


# Type alias for all document schemas
ExtractedDocument = Union[Invoice, ShippingOrder, InventoryReport, PurchaseOrder]


class DocumentExtractor:
    """
    Extracts structured data from documents using OpenAI LLM.
    
    Usage:
        extractor = DocumentExtractor()
        invoice = extractor.extract(text, DocumentType.INVOICE)
        print(invoice.invoice_number)
    """
    
    # System prompts for each document type
    SYSTEM_PROMPTS = {
        DocumentType.INVOICE: """You are a document extraction assistant. 
Extract structured data from the invoice document provided.
Be precise with numbers and dates. Use YYYY-MM-DD format for dates.
If a field is not found in the document, leave it as null.""",
        
        DocumentType.SHIPPING_ORDER: """You are a document extraction assistant.
Extract structured data from the shipping order document provided.
Be precise with tracking numbers and addresses. Use YYYY-MM-DD format for dates.
If a field is not found in the document, leave it as null.""",
        
        DocumentType.INVENTORY_REPORT: """You are a document extraction assistant.
Extract structured data from the inventory report document provided.
Be precise with quantities and SKUs. Use YYYY-MM-DD format for dates.
If a field is not found in the document, leave it as null.""",
        
        DocumentType.PURCHASE_ORDER: """You are a document extraction assistant.
Extract structured data from the purchase order document provided.
Be precise with PO numbers, quantities, and prices. Use YYYY-MM-DD format for dates.
If a field is not found in the document, leave it as null.""",
    }
    
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        """
        Initialize the extractor.
        
        Args:
            api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
            model: OpenAI model to use (default: gpt-4o-mini)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
    
    def extract(
        self, 
        text: str, 
        document_type: DocumentType
    ) -> tuple[ExtractedDocument, ValidationResult]:
        """
        Extract structured data from a document.
        
        Args:
            text: The document text to extract from
            document_type: The type of document (determines which schema to use)
            
        Returns:
            Tuple of (extracted_data, validation_result)
        """
        # Get the appropriate schema for this document type
        schema_class = get_schema_for_type(document_type)
        system_prompt = self.SYSTEM_PROMPTS[document_type]
        
        try:
            # Use OpenAI's structured output with Pydantic
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract data from this document:\n\n{text}"}
                ],
                response_format=schema_class,
            )
            
            # Get the parsed object
            extracted = response.choices[0].message.parsed
            
            # Validation passed (OpenAI + Pydantic already validated)
            validation = ValidationResult(is_valid=True, errors=[])
            
            return extracted, validation
            
        except ValidationError as e:
            # Pydantic validation failed
            errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            validation = ValidationResult(is_valid=False, errors=errors)
            
            # Return empty schema with errors
            return schema_class.model_construct(), validation
            
        except Exception as e:
            # Other errors (API errors, etc.)
            validation = ValidationResult(
                is_valid=False, 
                errors=[f"Extraction failed: {str(e)}"]
            )
            return schema_class.model_construct(), validation
    
    def extract_raw(self, text: str, document_type: DocumentType) -> dict:
        """
        Extract data and return as a raw dictionary (for JSON serialization).
        
        Args:
            text: The document text
            document_type: The document type
            
        Returns:
            Dictionary representation of extracted data
        """
        extracted, validation = self.extract(text, document_type)
        return {
            "data": extracted.model_dump(),
            "validation": validation.model_dump(),
        }


class MockExtractor:
    """
    Mock extractor for testing without API calls.
    
    Returns predefined responses for each document type.
    """
    
    MOCK_RESPONSES = {
        DocumentType.INVOICE: Invoice(
            invoice_number="INV-MOCK-001",
            date="2024-01-15",
            vendor="Mock Vendor Inc.",
            total_amount=1000.00,
            bill_to="Test Customer",
        ),
        DocumentType.SHIPPING_ORDER: ShippingOrder(
            tracking_number="MOCK123456789",
            ship_date="2024-01-15",
            carrier="MockEx",
            origin="123 Origin St, City, ST 12345",
            destination="456 Destination Ave, Town, ST 67890",
        ),
        DocumentType.INVENTORY_REPORT: InventoryReport(
            report_date="2024-01-15",
            warehouse="Warehouse A",
            items=[],
            total_units=0,
        ),
        DocumentType.PURCHASE_ORDER: PurchaseOrder(
            po_number="PO-MOCK-001",
            date="2024-01-15",
            vendor="Mock Supplier",
            items=[],
            total_value=500.00,
        ),
    }
    
    def extract(
        self, 
        text: str, 
        document_type: DocumentType
    ) -> tuple[ExtractedDocument, ValidationResult]:
        """Return mock data for testing."""
        extracted = self.MOCK_RESPONSES[document_type]
        validation = ValidationResult(is_valid=True, errors=[])
        return extracted, validation
    
    def extract_raw(self, text: str, document_type: DocumentType) -> dict:
        """Return mock data as dictionary."""
        extracted, validation = self.extract(text, document_type)
        return {
            "data": extracted.model_dump(),
            "validation": validation.model_dump(),
        }

