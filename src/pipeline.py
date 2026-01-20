"""
Document Processing Pipeline.

Orchestrates the full flow: classify → extract → validate.
This is the main entry point for processing documents.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .schemas import (
    DocumentType,
    ClassificationResult,
    ValidationResult,
    ProcessingResult,
    DOCUMENT_SCHEMAS,
)
from .classifier import DocumentClassifier, train_from_huggingface
from .extractor import DocumentExtractor, MockExtractor


@dataclass
class Document:
    """A document with its text and metadata."""
    text: str
    id: str = ""
    original_label: Optional[str] = None  # Ground truth label if available


class DocumentPipeline:
    """
    Complete document processing pipeline.
    
    Combines classification and extraction into a single interface.
    
    Usage:
        pipeline = DocumentPipeline()
        pipeline.load_classifier()  # Train or load classifier
        result = pipeline.process("INVOICE #123...")
    """
    
    def __init__(
        self,
        classifier: Optional[DocumentClassifier] = None,
        extractor: Optional[DocumentExtractor | MockExtractor] = None,
        use_mock_extractor: bool = False,
    ):
        """
        Initialize the pipeline.
        
        Args:
            classifier: Pre-trained classifier (or None to train later)
            extractor: Extractor instance (or None to create with env API key)
            use_mock_extractor: If True, use mock extractor (for testing)
        """
        self.classifier = classifier
        
        if extractor:
            self.extractor = extractor
        elif use_mock_extractor:
            self.extractor = MockExtractor()
        else:
            self.extractor = None  # Will be created on first use
    
    def load_classifier(
        self,
        model_path: Optional[Path] = None,
        train_sample_size: Optional[int] = None,
    ) -> dict:
        """
        Load or train the classifier.
        
        Args:
            model_path: Path to saved model (loads if exists, saves after training)
            train_sample_size: Limit training data size (for faster training)
            
        Returns:
            Training metrics if trained, empty dict if loaded
        """
        if model_path and model_path.exists():
            print(f"Loading classifier from {model_path}...")
            self.classifier = DocumentClassifier(model_path)
            return {}
        
        print("Training classifier from HuggingFace dataset...")
        self.classifier = train_from_huggingface(sample_size=train_sample_size)
        
        if model_path:
            print(f"Saving classifier to {model_path}...")
            self.classifier.save(model_path)
        
        return {"status": "trained"}
    
    def _ensure_extractor(self) -> None:
        """Create extractor if not initialized."""
        if self.extractor is None:
            self.extractor = DocumentExtractor()
    
    def classify(self, text: str) -> ClassificationResult:
        """
        Classify a document without extraction.
        
        Args:
            text: Document text
            
        Returns:
            ClassificationResult with type and confidence
        """
        if self.classifier is None:
            raise ValueError("Classifier not loaded. Call load_classifier() first.")
        
        return self.classifier.classify(text)
    
    def classify_batch(self, texts: list[str]) -> list[ClassificationResult]:
        """
        Classify multiple documents.
        
        Args:
            texts: List of document texts
            
        Returns:
            List of ClassificationResult objects
        """
        if self.classifier is None:
            raise ValueError("Classifier not loaded. Call load_classifier() first.")
        
        return self.classifier.classify_batch(texts)
    
    def extract(self, text: str, document_type: DocumentType) -> dict:
        """
        Extract data from a document (without classification).
        
        Args:
            text: Document text
            document_type: Known document type
            
        Returns:
            Dictionary with extracted data and validation result
        """
        self._ensure_extractor()
        return self.extractor.extract_raw(text, document_type)
    
    def process(self, text: str, skip_extraction: bool = False) -> ProcessingResult:
        """
        Process a document through the full pipeline.
        
        Steps:
        1. Classify the document
        2. Extract structured data using appropriate schema
        3. Validate the extraction
        
        Args:
            text: Document text
            skip_extraction: If True, only classify (faster, no API call)
            
        Returns:
            ProcessingResult with classification, extraction, and validation
        """
        # Step 1: Classify
        classification = self.classify(text)
        
        if skip_extraction:
            # Return with empty extraction
            return ProcessingResult(
                classification=classification,
                extracted_data={},
                validation=ValidationResult(is_valid=True, errors=["Extraction skipped"]),
                raw_text=text,
            )
        
        # Step 2: Extract
        self._ensure_extractor()
        extracted, validation = self.extractor.extract(text, classification.document_type)
        
        # Step 3: Return result
        return ProcessingResult(
            classification=classification,
            extracted_data=extracted.model_dump(),
            validation=validation,
            raw_text=text,
        )
    
    def process_batch(
        self, 
        texts: list[str], 
        skip_extraction: bool = True
    ) -> list[ProcessingResult]:
        """
        Process multiple documents.
        
        Note: By default, extraction is skipped for batch processing
        to avoid excessive API calls. Set skip_extraction=False to 
        extract all documents (will be slow and costly for large batches).
        
        Args:
            texts: List of document texts
            skip_extraction: Skip LLM extraction (default True for batch)
            
        Returns:
            List of ProcessingResult objects
        """
        # Batch classify for efficiency
        classifications = self.classify_batch(texts)
        
        results = []
        for text, classification in zip(texts, classifications):
            if skip_extraction:
                result = ProcessingResult(
                    classification=classification,
                    extracted_data={},
                    validation=ValidationResult(is_valid=True, errors=["Extraction skipped"]),
                    raw_text=text,
                )
            else:
                self._ensure_extractor()
                extracted, validation = self.extractor.extract(
                    text, classification.document_type
                )
                result = ProcessingResult(
                    classification=classification,
                    extracted_data=extracted.model_dump(),
                    validation=validation,
                    raw_text=text,
                )
            
            results.append(result)
        
        return results


def load_sample_documents(n_samples: int = 20) -> list[Document]:
    """
    Load sample documents from HuggingFace dataset.
    
    Args:
        n_samples: Number of samples to load
        
    Returns:
        List of Document objects
    """
    from datasets import load_dataset
    
    dataset = load_dataset("AyoubChLin/CompanyDocuments")
    train_data = dataset["train"].shuffle(seed=42).select(range(min(n_samples, len(dataset["train"]))))
    
    documents = []
    for i, item in enumerate(train_data):
        doc = Document(
            text=item["file_content"],  # Dataset uses 'file_content' not 'text'
            id=f"doc_{i+1}",
            original_label=item["document_type"],  # Dataset uses 'document_type' not 'label'
        )
        documents.append(doc)
    
    return documents

