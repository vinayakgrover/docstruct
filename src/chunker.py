"""
Text chunking for handling large documents.

Chunks documents into smaller pieces for:
- LLM context limits
- Better extraction accuracy on long docs
- Parallel processing
- Future RAG/search capabilities
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    """A chunk of text with position metadata."""
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    total_chunks: int


def chunk_text(
    text: str,
    max_chars: int = 4000,
    overlap: int = 200,
    min_chunk_size: int = 100,
) -> list[Chunk]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Document text to chunk
        max_chars: Maximum characters per chunk
        overlap: Character overlap between chunks (for context continuity)
        min_chunk_size: Don't create tiny trailing chunks
        
    Returns:
        List of Chunk objects with position metadata
    """
    # Short documents don't need chunking
    if len(text) <= max_chars:
        return [Chunk(
            text=text,
            chunk_index=0,
            start_char=0,
            end_char=len(text),
            total_chunks=1,
        )]
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        end = start + max_chars
        
        # Try to break at sentence/paragraph boundary
        if end < len(text):
            # Look for natural break point
            for sep in ['\n\n', '\n', '. ', ', ', ' ']:
                break_point = text.rfind(sep, start + max_chars // 2, end)
                if break_point != -1:
                    end = break_point + len(sep)
                    break
        
        chunk_text_content = text[start:end].strip()
        
        # Skip empty chunks
        if chunk_text_content:
            chunks.append(Chunk(
                text=chunk_text_content,
                chunk_index=chunk_index,
                start_char=start,
                end_char=end,
                total_chunks=-1,  # Will update after
            ))
            chunk_index += 1
        
        # Move start, accounting for overlap
        start = end - overlap if end < len(text) else end
        
        # Avoid tiny trailing chunks
        if len(text) - start < min_chunk_size:
            break
    
    # Update total_chunks count
    total = len(chunks)
    for chunk in chunks:
        chunk.total_chunks = total
    
    return chunks


def merge_chunk_extractions(extractions: list[dict]) -> dict:
    """
    Merge extractions from multiple chunks into a single result.
    
    For interview discussion: In production, you'd implement smarter
    merging logic (deduplication, conflict resolution, confidence weighting).
    
    Args:
        extractions: List of extraction dicts from each chunk
        
    Returns:
        Merged extraction dict
    """
    if not extractions:
        return {}
    
    if len(extractions) == 1:
        return extractions[0]
    
    # Simple merge: take first non-null value for each field
    merged = {}
    for extraction in extractions:
        for key, value in extraction.items():
            if key not in merged or merged[key] is None:
                merged[key] = value
            elif isinstance(value, list) and isinstance(merged[key], list):
                # Combine lists (e.g., line items from different chunks)
                merged[key].extend(value)
    
    return merged

