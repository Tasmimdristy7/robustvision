"""
Logging utilities for RobustVision
"""

import logging
import sys
from typing import Optional

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger instance for RobustVision
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (optional)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set level if provided
    if level is not None:
        logger.setLevel(level)
    
    # Add handler if none exists
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Set default level if not already set
        if level is None:
            logger.setLevel(logging.INFO)
    
    return logger

def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """
    Setup logging configuration for RobustVision
    
    Args:
        level: Logging level
        log_file: Optional log file path
    """
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    ) 