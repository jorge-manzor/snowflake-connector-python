# The Logging Package
Python's `logging` module provides a flexible framework for emitting log messages from applications. It's designed to be thread-safe and highly configurable.

## Key Concepts in Logging

1. **Loggers:** The entry points into the logging system. Loggers are organized in a hierarchy using dot notation (e.g., 'parent', 'parent.child').

2. **Handlers:** Determine what happens to log messages (e.g., write to a file, send to console, email).

3. **Formatters:** Define the structure and content of log messages.

4. **Levels:** Define the severity of messages:
   - DEBUG (10): Detailed information for debugging.
   - INFO (20): Confirmation that things are working as expected.
   - WARNING (30): Something unexpected happened, but the program is still working.
   - ERROR (40): A more serious problem prevented a function from working.
   - CRITICAL (50): A very serious error that might prevent the program from continuing.

## Our Logging Configuration
Let's analyze our `setup_logging` function:

```python
import logging
import sys

def setup_logging(level=logging.INFO):
    """Configure logging for the application."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log')
        ]
    )
    
    # Reduce verbosity of third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('snowflake.connector').setLevel(logging.WARNING)
```

This function does several important things:

**1. Sets the Basic Configuration:**
   - `level=level`: Sets the threshold for which messages will be processed (default is INFO)
   - `format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'`: Defines how log messages will look
     - `%(asctime)s`: Timestamp when the log was created
     - `%(name)s`: Logger name
     - `%(levelname)s`: Level of the message (DEBUG, INFO, etc.)
     - `%(message)s`: The actual log message

**2. Configures Multiple Handlers:**
   - `logging.StreamHandler(sys.stdout)`: Sends logs to the console(standard output)
   - `logging.FileHandler('app.log')`: Writes logs to a file named 'app.log'
  
  This means your logs will appear both in the console AND be saved to a file.

**3. Reduces Third-Party Logging Noise:**
   - Sets the log level for specific third-party libraries to WARNING.
   - This prevents your logs from being flooded with messages from libraries like urllib3 and snowflake.connector
   - Only warnings and more severe messages from these libraries will be logged.

## How to Use Logging in Your Code
Once configured, you can use logging throughout our application:
```python
import logging

# Get a logger for the current module
logger = logging.getLogger(__name__)

def some_function():
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    try:
        result = 10 / 0
    except Exception as e:
        logger.exception("An error occurred")  # Logs the error with traceback
```

The `__name__` variable gives each module its own logger, which helps identify where logs are coming from.