"""
Structured logging configuration for the oratory feedback system.

This module provides centralized logging configuration with consistent formatting,
log levels, and handlers for all components of the application.
"""

import logging
import sys
import os
from typing import Optional, Dict, Any
import json
from datetime import datetime


class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings for each log record.
    Useful for structured logging to be ingested by log analysis tools.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }
            
        # Add extra attributes
        if hasattr(record, "extra"):
            log_data["extra"] = record.extra
            
        return json.dumps(log_data)


def setup_logging(
    level: int = logging.INFO,
    json_output: bool = False,
    log_file: Optional[str] = None,
    module_levels: Optional[Dict[str, int]] = None
) -> None:
    """
    Configure application-wide logging settings.
    
    Args:
        level: Base log level (default: INFO)
        json_output: Whether to output logs as JSON (default: False)
        log_file: Optional file to write logs to
        module_levels: Dictionary mapping module names to specific log levels
    """
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    if json_output:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s:%(funcName)s:%(lineno)d] - %(message)s"
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific module log levels if provided
    if module_levels:
        for module, module_level in module_levels.items():
            logging.getLogger(module).setLevel(module_level)
    
    # Quiet noisy third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    logging.info("Logging system initialized")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name, typically __name__ of the module
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class LogContext:
    """
    Context manager for adding temporary context to log records.
    
    This class provides a way to add additional context to log records
    without modifying the logger's makeRecord method, which was causing issues.
    """
    
    def __init__(self, logger: logging.Logger, **context):
        """
        Initialize with logger and context variables.
        
        Args:
            logger: Logger to add context to
            **context: Key-value pairs to add to log context
        """
        self.logger = logger
        self.context = context
        self.original_log_methods = {}
    
    def __enter__(self):
        """Set up patched logging methods that include the context."""
        try:
            # Save original logging methods
            log_methods = ['debug', 'info', 'warning', 'error', 'critical', 'log']
            for method_name in log_methods:
                if hasattr(self.logger, method_name):
                    self.original_log_methods[method_name] = getattr(self.logger, method_name)
                    
                    # Create wrapper function to inject context
                    def make_log_wrapper(original_method):
                        def wrapper(msg, *args, **kwargs):
                            # Add our context to kwargs
                            kwargs.setdefault('extra', {}).update(self.context)
                            return original_method(msg, *args, **kwargs)
                        return wrapper
                    
                    # Apply the wrapper
                    setattr(self.logger, method_name, make_log_wrapper(self.original_log_methods[method_name]))
        
        except Exception as e:
            print(f"Warning: Could not setup LogContext: {str(e)}")
        
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original logging methods."""
        try:
            # Restore original methods
            for method_name, original_method in self.original_log_methods.items():
                setattr(self.logger, method_name, original_method)
        except Exception as e:
            print(f"Warning: Error during LogContext cleanup: {str(e)}")


def log_execution_time(logger: logging.Logger, level: int = logging.INFO):
    """
    Decorator to log execution time of a function.
    
    Args:
        logger: Logger to use
        level: Log level to use
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                elapsed = datetime.now() - start_time
                logger.log(
                    level,
                    f"Function {func.__name__} executed in {elapsed.total_seconds():.3f} seconds",
                    extra={"execution_time": elapsed.total_seconds()}
                )
                return result
            except Exception as e:
                elapsed = datetime.now() - start_time
                logger.error(
                    f"Function {func.__name__} failed after {elapsed.total_seconds():.3f} seconds: {str(e)}",
                    exc_info=True,
                    extra={"execution_time": elapsed.total_seconds()}
                )
                raise
        return wrapper
    return decorator
