

### Context:
A Fortune 500 customer has a document lake containing thousands of unstructured business documents — shipping orders, invoices, inventory reports, and purchase orders. They need to automatically classify incoming documents and extract structured metadata so it can flow into their Collibra data catalog.

### Your task:
Build a document processing pipeline in Python that:

1. **Classifies** a document into one of four types:
   - `shipping_order`
   - `invoice`
   - `inventory_report`
   - `purchase_order`

2. **Extracts** structured data from document text into a standardized JSON schema

