"""
Utility Functions Module

This module provides various utility functions for the NeuroCompressor project.
"""

import os
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path to write logs to
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger.info(f"Logging configured at level: {log_level}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif path.suffix == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_results(results: Dict[str, Any], output_dir: str, filename: Optional[str] = None):
    """
    Save experiment results to a file.
    
    Args:
        results: Dictionary containing results
        output_dir: Directory to save results
        filename: Optional filename (default: timestamp-based)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.json"
    
    filepath = output_path / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved results to {filepath}")


def ensure_dir(directory: str):
    """Ensure a directory exists, creating it if necessary."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def read_text_file(filepath: str) -> str:
    """Read text from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def write_text_file(filepath: str, content: str):
    """Write text to a file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


def get_file_size(filepath: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(filepath)


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

