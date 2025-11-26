"""
NeuroCompressor: A Large Language Model-based Text Compression Framework

This package provides tools for compressing and decompressing text using
large language models for research purposes.
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from neurocompressor.compressor import TextCompressor
from neurocompressor.decompressor import TextDecompressor
from neurocompressor.models import LLMModel

__all__ = ["TextCompressor", "TextDecompressor", "LLMModel"]

