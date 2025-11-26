"""
Unit tests for TextCompressor
"""

import unittest
from unittest.mock import Mock, patch
from neurocompressor import TextCompressor, LLMModel


class TestTextCompressor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock model
        self.mock_model = Mock(spec=LLMModel)
        self.mock_model.model_name = "test-model"
        self.mock_model.generate.return_value = "Compressed text"
        self.mock_model.encode.return_value = "encoded"
        self.mock_model.decode.return_value = "decoded"
        
        self.compressor = TextCompressor(
            model=self.mock_model,
            compression_strategy="semantic",
            chunk_size=100,
            overlap=10
        )
    
    def test_initialization(self):
        """Test compressor initialization."""
        self.assertEqual(self.compressor.compression_strategy, "semantic")
        self.assertEqual(self.compressor.chunk_size, 100)
        self.assertEqual(self.compressor.overlap, 10)
    
    def test_compress_semantic(self):
        """Test semantic compression."""
        text = "This is a test text that needs to be compressed."
        result = self.compressor.compress(text, compression_ratio=0.5)
        
        self.assertIn("compressed", result)
        self.assertIn("original_length", result)
        self.assertIn("compressed_length", result)
        self.assertIn("compression_ratio", result)
        self.assertEqual(result["strategy"], "semantic")
    
    def test_chunk_text(self):
        """Test text chunking."""
        text = "A" * 250  # 250 characters
        chunks = self.compressor._chunk_text(text)
        
        self.assertGreater(len(chunks), 1)
        # Check that chunks respect size and overlap
        for chunk in chunks:
            self.assertLessEqual(len(chunk), self.compressor.chunk_size)
    
    def test_compress_prompt_creation(self):
        """Test compression prompt creation."""
        text = "Test text"
        prompt = self.compressor._create_compression_prompt(text, 0.5, True)
        
        self.assertIn(text, prompt)
        self.assertIn("Compress", prompt)
        self.assertIn("50", prompt)  # 50% compression ratio


if __name__ == "__main__":
    unittest.main()

