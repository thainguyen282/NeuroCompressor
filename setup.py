"""
Setup script for NeuroCompressor
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="neurocompressor",
    version="0.1.0",
    author="Research Team",
    author_email="research@example.com",
    description="A Large Language Model-based Text Compression Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/NeuroCompressor",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "neurocompress=experiments.run_compression_experiment:main",
        ],
    },
)

