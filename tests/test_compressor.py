"""
Unit tests for TextCompressor
"""

import unittest
from unittest.mock import Mock, MagicMock
import torch
from neurocompressor import TextCompressor


class TestTextCompressor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock model and tokenizer
        self.mock_model = MagicMock()
        self.mock_model.device = torch.device('cpu')
        self.mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        self.mock_model.parameters.return_value = [torch.tensor([1.0])]
        
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.encode.return_value = [1, 2, 3]
        self.mock_tokenizer.decode.return_value = "decoded text"
        # Make tokenizer callable (when called with text, returns tokenized result)
        tokenized_result = type('obj', (object,), {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5, 6]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1]])
        })()
        self.mock_tokenizer.return_value = tokenized_result
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.eos_token_id = 2
        self.mock_tokenizer.bos_token_id = None
        self.mock_tokenizer.cls_token_id = None
        self.mock_tokenizer.sep_token_id = None
        self.mock_tokenizer.unk_token_id = 1
        self.mock_tokenizer.chat_template = None
        self.mock_tokenizer.apply_chat_template = None
        
        self.compressor = TextCompressor(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            compression_strategy="tree",
            chunk_size=100,
            overlap=10
        )
    
    def test_initialization(self):
        """Test compressor initialization."""
        self.assertEqual(self.compressor.compression_strategy, "tree")
        self.assertEqual(self.compressor.chunk_size, 100)
        self.assertEqual(self.compressor.overlap, 10)
    
    def test_compress_tree(self):
        """Test tree compression."""
        # Setup tokenizer mock to return proper format
        self.mock_tokenizer.return_value = type('obj', (object,), {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5, 6]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1]])
        })()
        
        text = "This is a test text that needs to be compressed."
        result = self.compressor.compress(text, compression_ratio=0.5)
        
        self.assertIn("compressed_v1", result)
        self.assertIn("compression_ratio_v1", result)
    
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

