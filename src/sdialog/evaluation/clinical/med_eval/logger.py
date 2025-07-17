# medical_dialogue_evaluator/logger.py
"""
Configures and provides a centralized logger for the application.
"""
import logging
import sys

def get_logger(name="med_eval"):
    """Initializes a standardized logger."""
    logger = logging.getLogger(name)
    
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger

logger = get_logger()
