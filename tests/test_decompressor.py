"""
Unit tests for TextDecompressor
"""

import unittest
from unittest.mock import Mock
from neurocompressor import TextDecompressor, LLMModel


class TestTextDecompressor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = Mock(spec=LLMModel)
        self.mock_model.model_name = "test-model"
        self.mock_model.generate.return_value = "Decompressed text"
        
        self.decompressor = TextDecompressor(
            model=self.mock_model,
            decompression_strategy="semantic"
        )
    
    def test_initialization(self):
        """Test decompressor initialization."""
        self.assertEqual(self.decompressor.decompression_strategy, "semantic")
    
    def test_decompress_semantic(self):
        """Test semantic decompression."""
        compressed_data = {
            "compressed": "Compressed text",
            "strategy": "semantic",
            "original_length": 100,
            "metadata": {}
        }
        
        result = self.decompressor.decompress(compressed_data)
        
        self.assertIsInstance(result, str)
        self.mock_model.generate.assert_called_once()
    
    def test_decompression_prompt_creation(self):
        """Test decompression prompt creation."""
        compressed_text = "Compressed"
        prompt = self.decompressor._create_decompression_prompt(
            compressed_text, 100, True
        )
        
        self.assertIn(compressed_text, prompt)
        self.assertIn("Expand", prompt)


if __name__ == "__main__":
    unittest.main()

