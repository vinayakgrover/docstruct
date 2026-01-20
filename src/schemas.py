"""
Pydantic schemas for document extraction.

Each document type has its own schema defining the fields to extract.
These schemas are passed to the LLM for structured output extraction.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class DocumentType(str, Enum):
    """The four document types we classify."""
    INVOICE = "invoice"
    SHIPPING_ORDER = "shipping_order"
    INVENTORY_REPORT = "inventory_report"
    PURCHASE_ORDER = "purchase_order"


# ============================================================================
# Shared Models
# ============================================================================

class LineItem(BaseModel):
    """A line item in an invoice or purchase order."""
    description: str = Field(..., description="Description of the item")
    quantity: int = Field(..., description="Quantity ordered/purchased")
    unit_price: float = Field(..., description="Price per unit")
    total: Optional[float] = Field(None, description="Line total (quantity × unit_price)")


class InventoryItem(BaseModel):
    """An item in an inventory report."""
    item_name: str = Field(..., description="Name of the inventory item")
    sku: Optional[str] = Field(None, description="Stock Keeping Unit / Item ID")
    quantity_on_hand: int = Field(..., description="Current quantity in stock")
    unit_cost: Optional[float] = Field(None, description="Cost per unit")


# ============================================================================
# Document Schemas
# ============================================================================

class Invoice(BaseModel):
    """Schema for invoice documents."""
    invoice_number: str = Field(..., description="Unique invoice identifier (e.g., INV-2024-001)")
    date: str = Field(..., description="Invoice date in YYYY-MM-DD format")
    vendor: str = Field(..., description="Company or person issuing the invoice")
    bill_to: Optional[str] = Field(None, description="Company or person being billed")
    total_amount: float = Field(..., description="Total invoice amount")
    currency: Optional[str] = Field("USD", description="Currency code (e.g., USD, EUR)")
    due_date: Optional[str] = Field(None, description="Payment due date in YYYY-MM-DD format")
    line_items: Optional[list[LineItem]] = Field(None, description="List of items/services")
    notes: Optional[str] = Field(None, description="Additional notes or terms")


class ShippingOrder(BaseModel):
    """Schema for shipping order documents."""
    tracking_number: str = Field(..., description="Shipment tracking number")
    ship_date: str = Field(..., description="Ship date in YYYY-MM-DD format")
    carrier: str = Field(..., description="Shipping carrier (e.g., UPS, FedEx, DHL)")
    origin: str = Field(..., description="Origin address or location")
    destination: str = Field(..., description="Destination/delivery address")
    recipient: Optional[str] = Field(None, description="Recipient name")
    weight: Optional[float] = Field(None, description="Package weight in lbs or kg")
    dimensions: Optional[str] = Field(None, description="Package dimensions (L×W×H)")
    status: Optional[str] = Field(None, description="Current shipment status")
    delivery_date: Optional[str] = Field(None, description="Expected/actual delivery date")


class InventoryReport(BaseModel):
    """Schema for inventory report documents."""
    report_date: str = Field(..., description="Report date in YYYY-MM-DD format")
    report_id: Optional[str] = Field(None, description="Unique report identifier")
    warehouse: str = Field(..., description="Warehouse location or identifier")
    items: list[InventoryItem] = Field(..., description="List of inventory items")
    total_units: int = Field(..., description="Total units across all items")
    total_value: Optional[float] = Field(None, description="Total inventory value")
    notes: Optional[str] = Field(None, description="Additional notes or observations")


class PurchaseOrder(BaseModel):
    """Schema for purchase order documents."""
    po_number: str = Field(..., description="Purchase order number (e.g., PO-2024-001)")
    date: str = Field(..., description="Order date in YYYY-MM-DD format")
    vendor: str = Field(..., description="Supplier/vendor name")
    buyer: Optional[str] = Field(None, description="Purchasing company/person")
    items: list[LineItem] = Field(..., description="List of items ordered")
    total_value: float = Field(..., description="Total order value")
    currency: Optional[str] = Field("USD", description="Currency code")
    ship_to: Optional[str] = Field(None, description="Delivery address")
    terms: Optional[str] = Field(None, description="Payment/delivery terms")
    notes: Optional[str] = Field(None, description="Additional notes")


# ============================================================================
# Classification Result
# ============================================================================

class ClassificationResult(BaseModel):
    """Result from the document classifier."""
    document_type: DocumentType
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")


class ValidationResult(BaseModel):
    """Result from schema validation."""
    is_valid: bool
    errors: list[str] = Field(default_factory=list)


class ProcessingResult(BaseModel):
    """Complete result from processing a document."""
    classification: ClassificationResult
    extracted_data: dict  # Will be one of: Invoice, ShippingOrder, InventoryReport, PurchaseOrder
    validation: ValidationResult
    raw_text: Optional[str] = Field(None, description="Original document text")


# ============================================================================
# Schema Mapping
# ============================================================================

DOCUMENT_SCHEMAS = {
    DocumentType.INVOICE: Invoice,
    DocumentType.SHIPPING_ORDER: ShippingOrder,
    DocumentType.INVENTORY_REPORT: InventoryReport,
    DocumentType.PURCHASE_ORDER: PurchaseOrder,
}


def get_schema_for_type(doc_type: DocumentType) -> type[BaseModel]:
    """Get the Pydantic schema class for a document type."""
    return DOCUMENT_SCHEMAS[doc_type]

