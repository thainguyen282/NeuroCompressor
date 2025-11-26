"""
Unit tests for metrics calculation
"""

import unittest
from neurocompressor.metrics import calculate_metrics, _tokenize, _calculate_character_overlap


class TestMetrics(unittest.TestCase):
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        original = "This is a test text."
        reconstructed = "This is a test text."
        
        metrics = calculate_metrics(original, reconstructed)
        
        self.assertIn("compression_ratio", metrics)
        self.assertIn("length_ratio", metrics)
        self.assertIn("character_overlap", metrics)
        self.assertIn("word_overlap", metrics)
    
    def test_tokenize(self):
        """Test text tokenization."""
        text = "This is a test."
        tokens = _tokenize(text)
        
        self.assertEqual(len(tokens), 4)
        self.assertIn("this", tokens)
        self.assertIn("is", tokens)
    
    def test_character_overlap(self):
        """Test character overlap calculation."""
        text1 = "abc"
        text2 = "bcd"
        
        overlap = _calculate_character_overlap(text1, text2)
        
        self.assertGreater(overlap, 0)
        self.assertLessEqual(overlap, 1)


if __name__ == "__main__":
    unittest.main()

