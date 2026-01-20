"""
Document Classifier using TF-IDF + Logistic Regression.

This module provides fast, offline document classification without requiring
any API calls. The classifier is trained on the HuggingFace CompanyDocuments
dataset and can classify documents into 4 types.

Why TF-IDF + Logistic Regression?
- Fast inference (milliseconds per document)
- No API costs
- Works offline
- Interpretable (can see which words drive decisions)
- Easy to retrain with new data
"""

import pickle
from pathlib import Path
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

from .schemas import ClassificationResult, DocumentType


class DocumentClassifier:
    """
    Classifies documents into one of four types using TF-IDF + Logistic Regression.
    
    Usage:
        classifier = DocumentClassifier()
        classifier.train(texts, labels)  # Train on data
        result = classifier.classify("INVOICE #123...")  # Classify new doc
    """
    
    # Map dataset labels to our DocumentType enum
    # The HuggingFace dataset uses different label names
    LABEL_MAP = {
        "invoice": DocumentType.INVOICE,
        "invoices": DocumentType.INVOICE,
        "Invoices": DocumentType.INVOICE,
        "shipping_order": DocumentType.SHIPPING_ORDER,
        "shipping_orders": DocumentType.SHIPPING_ORDER,
        "Shipping Orders": DocumentType.SHIPPING_ORDER,
        "inventory_report": DocumentType.INVENTORY_REPORT,
        "purchase_order": DocumentType.PURCHASE_ORDER,
        "purchase_orders": DocumentType.PURCHASE_ORDER,
        "Purchase Orders": DocumentType.PURCHASE_ORDER,
    }
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the classifier.
        
        Args:
            model_path: Optional path to load a pre-trained model from.
        """
        self.pipeline: Optional[Pipeline] = None
        self.is_trained = False
        
        if model_path and model_path.exists():
            self.load(model_path)
    
    def train(self, texts: list[str], labels: list[str], test_size: float = 0.2) -> dict:
        """
        Train the classifier on document texts and labels.
        
        Args:
            texts: List of document texts
            labels: List of document type labels
            test_size: Fraction of data to use for testing (default 0.2)
            
        Returns:
            Dictionary with training metrics
        """
        # Create the ML pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,      # Limit vocabulary size
                ngram_range=(1, 2),     # Use unigrams and bigrams
                stop_words='english',   # Remove common words
                min_df=2,               # Ignore very rare terms
                max_df=0.95,            # Ignore terms in >95% of docs
            )),
            ('classifier', LogisticRegression(
                max_iter=1000,
                class_weight='balanced',  # Handle imbalanced classes
                random_state=42,
            ))
        ])
        
        # Split data for training and evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Train the model
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on test set
        y_pred = self.pipeline.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        # Get detailed metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            "accuracy": accuracy,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "classification_report": report,
        }
    
    def classify(self, text: str) -> ClassificationResult:
        """
        Classify a single document.
        
        Args:
            text: The document text to classify
            
        Returns:
            ClassificationResult with document_type and confidence
            
        Raises:
            ValueError: If the classifier hasn't been trained
        """
        if not self.is_trained or self.pipeline is None:
            raise ValueError("Classifier has not been trained. Call train() first.")
        
        # Get prediction and probability
        label = self.pipeline.predict([text])[0]
        probabilities = self.pipeline.predict_proba([text])[0]
        
        # Get confidence for the predicted class
        class_idx = list(self.pipeline.classes_).index(label)
        confidence = float(probabilities[class_idx])
        
        # Map label to DocumentType
        doc_type = self.LABEL_MAP.get(label, DocumentType.INVOICE)
        
        return ClassificationResult(
            document_type=doc_type,
            confidence=confidence
        )
    
    def classify_batch(self, texts: list[str]) -> list[ClassificationResult]:
        """
        Classify multiple documents at once (more efficient).
        
        Args:
            texts: List of document texts
            
        Returns:
            List of ClassificationResult objects
        """
        if not self.is_trained or self.pipeline is None:
            raise ValueError("Classifier has not been trained. Call train() first.")
        
        labels = self.pipeline.predict(texts)
        probabilities = self.pipeline.predict_proba(texts)
        
        results = []
        for i, (label, probs) in enumerate(zip(labels, probabilities)):
            class_idx = list(self.pipeline.classes_).index(label)
            confidence = float(probs[class_idx])
            doc_type = self.LABEL_MAP.get(label, DocumentType.INVOICE)
            
            results.append(ClassificationResult(
                document_type=doc_type,
                confidence=confidence
            ))
        
        return results
    
    def get_top_features(self, n: int = 10) -> dict[str, list[str]]:
        """
        Get the top N most important features (words) for each class.
        
        Useful for understanding what the model learned.
        
        Args:
            n: Number of top features to return per class
            
        Returns:
            Dictionary mapping class labels to their top features
        """
        if not self.is_trained or self.pipeline is None:
            raise ValueError("Classifier has not been trained. Call train() first.")
        
        tfidf = self.pipeline.named_steps['tfidf']
        clf = self.pipeline.named_steps['classifier']
        
        feature_names = tfidf.get_feature_names_out()
        
        top_features = {}
        for i, class_label in enumerate(clf.classes_):
            # Get coefficients for this class
            if len(clf.classes_) == 2:
                # Binary classification
                coef = clf.coef_[0] if i == 1 else -clf.coef_[0]
            else:
                # Multi-class
                coef = clf.coef_[i]
            
            # Get top N features
            top_indices = np.argsort(coef)[-n:][::-1]
            top_features[class_label] = [feature_names[idx] for idx in top_indices]
        
        return top_features
    
    def save(self, path: Path) -> None:
        """Save the trained model to disk."""
        if not self.is_trained or self.pipeline is None:
            raise ValueError("Cannot save untrained model.")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f)
    
    def load(self, path: Path) -> None:
        """Load a trained model from disk."""
        with open(path, 'rb') as f:
            self.pipeline = pickle.load(f)
        self.is_trained = True


def train_from_huggingface(sample_size: Optional[int] = None) -> DocumentClassifier:
    """
    Train a classifier using the HuggingFace CompanyDocuments dataset.
    
    Args:
        sample_size: Optional limit on number of samples to use (for faster training)
        
    Returns:
        Trained DocumentClassifier instance
    """
    from datasets import load_dataset
    
    # Load dataset
    print("Loading HuggingFace dataset...")
    dataset = load_dataset("AyoubChLin/CompanyDocuments")
    train_data = dataset["train"]
    
    # Extract texts and labels
    # Note: This dataset uses 'file_content' and 'document_type' columns
    if sample_size:
        train_data = train_data.shuffle(seed=42).select(range(min(sample_size, len(train_data))))
    
    texts = train_data["file_content"]
    labels = train_data["document_type"]
    
    print(f"Training on {len(texts)} documents...")
    
    # Train classifier
    classifier = DocumentClassifier()
    metrics = classifier.train(texts, labels)
    
    print(f"Training complete! Accuracy: {metrics['accuracy']:.2%}")
    
    return classifier

