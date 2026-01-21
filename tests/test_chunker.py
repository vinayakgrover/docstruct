"""Tests for the text chunking module."""

import pytest
from src.chunker import chunk_text, merge_chunk_extractions, Chunk


class TestChunkText:
    """Tests for chunk_text function."""
    
    def test_short_text_single_chunk(self):
        """Short text should return a single chunk."""
        text = "This is a short document."
        chunks = chunk_text(text, max_chars=1000)
        
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].chunk_index == 0
        assert chunks[0].total_chunks == 1
    
    def test_long_text_multiple_chunks(self):
        """Long text should be split into multiple chunks."""
        # Create a long document
        text = "This is a sentence. " * 100  # ~2000 chars
        chunks = chunk_text(text, max_chars=500, overlap=50)
        
        assert len(chunks) > 1
        # First chunk should start at 0
        assert chunks[0].start_char == 0
        # All chunks should have correct total
        for chunk in chunks:
            assert chunk.total_chunks == len(chunks)
    
    def test_overlap_preserved(self):
        """Chunks should have overlapping content for context."""
        text = "Word " * 200  # 1000 chars
        chunks = chunk_text(text, max_chars=300, overlap=50)
        
        # Check that chunks overlap
        if len(chunks) > 1:
            # End of first chunk should overlap with start of second
            first_end = chunks[0].end_char
            second_start = chunks[1].start_char
            assert first_end > second_start  # Overlap exists
    
    def test_breaks_at_sentence_boundary(self):
        """Should prefer breaking at sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_text(text, max_chars=40, overlap=5)
        
        # Chunks should end at sentence boundaries when possible
        for chunk in chunks[:-1]:  # All but last
            # Should end with period and space or just the text
            assert chunk.text.endswith('.') or chunk.text.endswith('. ')
    
    def test_empty_text(self):
        """Empty text should return single empty chunk."""
        chunks = chunk_text("", max_chars=1000)
        assert len(chunks) == 1
        assert chunks[0].text == ""
    
    def test_chunk_metadata_correct(self):
        """Chunk metadata should be accurate."""
        text = "Hello world. " * 50
        chunks = chunk_text(text, max_chars=200, overlap=20)
        
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.start_char >= 0
            assert chunk.end_char <= len(text) + 100  # Allow for boundary adjustment
            assert chunk.total_chunks == len(chunks)


class TestMergeChunkExtractions:
    """Tests for merge_chunk_extractions function."""
    
    def test_single_extraction(self):
        """Single extraction should return as-is."""
        extraction = {"field1": "value1", "field2": "value2"}
        result = merge_chunk_extractions([extraction])
        assert result == extraction
    
    def test_empty_list(self):
        """Empty list should return empty dict."""
        result = merge_chunk_extractions([])
        assert result == {}
    
    def test_merge_non_overlapping_fields(self):
        """Non-overlapping fields should all be preserved."""
        extractions = [
            {"field1": "value1"},
            {"field2": "value2"},
        ]
        result = merge_chunk_extractions(extractions)
        assert result["field1"] == "value1"
        assert result["field2"] == "value2"
    
    def test_first_non_null_wins(self):
        """First non-null value should be used for conflicts."""
        extractions = [
            {"field1": "first_value"},
            {"field1": "second_value"},
        ]
        result = merge_chunk_extractions(extractions)
        assert result["field1"] == "first_value"
    
    def test_null_values_skipped(self):
        """Null values should be replaced by later non-null values."""
        extractions = [
            {"field1": None, "field2": "value2"},
            {"field1": "value1", "field2": None},
        ]
        result = merge_chunk_extractions(extractions)
        assert result["field1"] == "value1"
        assert result["field2"] == "value2"
    
    def test_lists_combined(self):
        """List fields should be combined."""
        extractions = [
            {"items": [{"name": "item1"}]},
            {"items": [{"name": "item2"}]},
        ]
        result = merge_chunk_extractions(extractions)
        assert len(result["items"]) == 2
        assert result["items"][0]["name"] == "item1"
        assert result["items"][1]["name"] == "item2"

