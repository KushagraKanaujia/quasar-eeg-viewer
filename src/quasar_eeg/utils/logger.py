"""
Logging utilities for Quasar EEG Viewer

Provides consistent logging across the application.
"""

import logging
import sys
from typing import Optional
from pathlib import Path


class Logger:
    """Centralized logging management"""

    _loggers = {}

    @classmethod
    def get_logger(
        cls,
        name: str,
        level: str = 'INFO',
        log_file: Optional[Path] = None
    ) -> logging.Logger:
        """
        Get or create a logger instance

        Args:
            name: Logger name (typically __name__ of the module)
            level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            log_file: Optional file path for logging output

        Returns:
            Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))

        # Prevent duplicate handlers
        if logger.handlers:
            return logger

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        cls._loggers[name] = logger
        return logger

    @classmethod
    def set_level(cls, name: str, level: str):
        """Change logging level for a specific logger"""
        if name in cls._loggers:
            cls._loggers[name].setLevel(getattr(logging, level.upper()))

    @classmethod
    def clear_loggers(cls):
        """Clear all logger instances"""
        cls._loggers.clear()
