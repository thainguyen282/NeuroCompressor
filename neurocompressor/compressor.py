"""
Text Compression Module

This module implements text compression using large language models.
"""

import logging
from typing import Optional, Dict, Any, List
import json
import hashlib

from neurocompressor.models import LLMModel

logger = logging.getLogger(__name__)


class TextCompressor:
    """
    Compresses text using LLM-based techniques.
    
    The compressor uses LLMs to create compact representations of text
    that can be later decompressed with high fidelity.
    """
    
    def __init__(self, model: LLMModel, compression_strategy: str = "semantic",
                 chunk_size: int = 1000, overlap: int = 100):
        """
        Initialize the text compressor.
        
        Args:
            model: LLM model instance for compression
            compression_strategy: Strategy to use ('semantic', 'token', 'hybrid')
            chunk_size: Size of text chunks for processing
            overlap: Overlap between chunks to maintain context
        """
        self.model = model
        self.compression_strategy = compression_strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        logger.info(f"Initialized TextCompressor with strategy: {compression_strategy}")
    
    def compress(self, text: str, compression_ratio: float = 0.5,
                 preserve_format: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Compress text using the specified strategy.
        
        Args:
            text: Input text to compress
            compression_ratio: Target compression ratio (0.0 to 1.0)
            preserve_format: Whether to preserve formatting information
            **kwargs: Additional compression parameters
            
        Returns:
            Dictionary containing compressed representation and metadata
        """
        logger.info(f"Compressing text of length {len(text)} characters")
        
        # if self.compression_strategy == "semantic":
        #     return self._compress_semantic(text, compression_ratio, preserve_format, **kwargs)
        # elif self.compression_strategy == "token":
        #     return self._compress_token(text, compression_ratio, preserve_format, **kwargs)
        # elif self.compression_strategy == "hybrid":
        #     return self._compress_hybrid(text, compression_ratio, preserve_format, **kwargs)
        # else:
        #     raise ValueError(f"Unknown compression strategy: {self.compression_strategy}")
    
    def _compress_semantic(self, text: str, compression_ratio: float,
                          preserve_format: bool, **kwargs) -> Dict[str, Any]:
        """
        Semantic compression: Uses LLM to create a semantic summary.
        """
        # Split text into chunks if too long
        chunks = self._chunk_text(text)
        compressed_chunks = []
        
        for chunk in chunks:
            prompt = self._create_compression_prompt(chunk, compression_ratio, preserve_format)
            compressed = self.model.generate(
                prompt,
                max_tokens=int(len(chunk) * compression_ratio),
                temperature=0.3,  # Lower temperature for more consistent compression
                **kwargs
            )
            compressed_chunks.append(compressed)
        
        compressed_text = " ".join(compressed_chunks)
        
        return {
            "compressed": compressed_text,
            "original_length": len(text),
            "compressed_length": len(compressed_text),
            "compression_ratio": len(compressed_text) / len(text) if len(text) > 0 else 0,
            "strategy": "semantic",
            "chunks": len(chunks),
            "metadata": {
                "preserve_format": preserve_format,
                "model": self.model.model_name,
                "hash": hashlib.md5(text.encode()).hexdigest()
            }
        }
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap
        
        return chunks
    
    def _create_compression_prompt(self, text: str, compression_ratio: float,
                                  preserve_format: bool) -> str:
        """Create a prompt for LLM-based compression."""
        format_instruction = "Preserve the original formatting, structure, and style." if preserve_format else ""
        
        prompt = f"""Compress the following text to approximately {compression_ratio*100:.0f}% of its original length while maintaining all key information and meaning. {format_instruction}

Original text:
{text}

Compressed text:"""
        
        return prompt
    
    def save(self, compressed_data: Dict[str, Any], filepath: str):
        """Save compressed data to a file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(compressed_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved compressed data to {filepath}")
    
    def load(self, filepath: str) -> Dict[str, Any]:
        """Load compressed data from a file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded compressed data from {filepath}")
        return data

