"""
Evaluation Metrics Module

This module provides metrics for evaluating compression and decompression quality.
"""

import logging
from typing import Dict, List
import re

logger = logging.getLogger(__name__)


def calculate_metrics(original: str, reconstructed: str) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for text reconstruction quality.
    
    Args:
        original: Original text
        reconstructed: Reconstructed text
        
    Returns:
        Dictionary containing various quality metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics["compression_ratio"] = len(reconstructed) / len(original) if len(original) > 0 else 0
    metrics["length_difference"] = abs(len(original) - len(reconstructed))
    metrics["length_ratio"] = len(reconstructed) / len(original) if len(original) > 0 else 0
    
    # Character-level metrics
    metrics["character_overlap"] = _calculate_character_overlap(original, reconstructed)
    
    # Word-level metrics
    original_words = _tokenize(original)
    reconstructed_words = _tokenize(reconstructed)
    metrics["word_overlap"] = _calculate_word_overlap(original_words, reconstructed_words)
    metrics["word_count_ratio"] = len(reconstructed_words) / len(original_words) if len(original_words) > 0 else 0
    
    # Semantic similarity (requires external library)
    try:
        metrics["semantic_similarity"] = _calculate_semantic_similarity(original, reconstructed)
    except ImportError:
        logger.warning("Semantic similarity calculation requires sentence-transformers. Install with: pip install sentence-transformers")
        metrics["semantic_similarity"] = 0.0
    
    # BLEU score (requires external library)
    try:
        metrics["bleu_score"] = _calculate_bleu(original, reconstructed)
    except ImportError:
        logger.warning("BLEU score calculation requires nltk. Install with: pip install nltk")
        metrics["bleu_score"] = 0.0
    
    return metrics


def _tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    # Simple word tokenization
    words = re.findall(r'\b\w+\b', text.lower())
    return words


def _calculate_character_overlap(text1: str, text2: str) -> float:
    """Calculate character-level overlap between two texts."""
    set1 = set(text1.lower())
    set2 = set(text2.lower())
    
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def _calculate_word_overlap(words1: List[str], words2: List[str]) -> float:
    """Calculate word-level overlap between two word lists."""
    set1 = set(words1)
    set2 = set(words2)
    
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def _calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity using sentence transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        
        # Use a lightweight model for semantic similarity
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode([text1, text2])
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return float(similarity)
    except ImportError:
        return 0.0


def _calculate_bleu(reference: str, candidate: str) -> float:
    """Calculate BLEU score between reference and candidate texts."""
    try:
        from nltk.translate.bleu_score import sentence_bleu
        from nltk.tokenize import word_tokenize
        
        reference_tokens = [word_tokenize(reference.lower())]
        candidate_tokens = word_tokenize(candidate.lower())
        
        score = sentence_bleu(reference_tokens, candidate_tokens)
        return float(score)
    except ImportError:
        return 0.0


def print_metrics(metrics: Dict[str, float]):
    """Pretty print metrics dictionary."""
    print("\n" + "="*50)
    print("Evaluation Metrics")
    print("="*50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:.4f}")
        else:
            print(f"{key:25s}: {value}")
    print("="*50 + "\n")

