import argparse
import logging
from utils.logging_config import setup_logging
from examples.holiday_data_example import process_year_range
from examples.query_to_pandas_string_example import run_query_to_pandas_string_example
from examples.query_to_pandas_file_example import run_query_to_pandas_file_example
from examples.cursor_to_pandas_string_example import run_cursor_to_pandas_string_example
from examples.cursor_to_pandas_file_example import run_cursor_to_pandas_file_example


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Snowflake connector examples')
    parser.add_argument('--example', choices=['holiday', 'query', 'all'], default='all',
                        help='Which example to run (default: all)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                        help='Set the logging level (default: INFO)')
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(level=log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Snowflake connector examples")
    
    # Run selected examples
    # if args.example in ['holiday', 'all']:
    #     logger.info("Running holiday data example")
    #     process_year_range()
    
    # if args.example in ['query', 'all']:
    #     logger.info("Running query from a string example")
    #     results = run_query_to_pandas_string_example()
    #     if not results.empty:
    #         logger.info(f"Query returned {len(results)} rows")
    
    # if args.example in ['query', 'all']:
    #     logger.info("Running query from a .sql file example")
    #     results = run_query_to_pandas_file_example()
    #     if not results.empty:
    #         logger.info(f"Query returned {len(results)} rows")
    
    # if args.example in ['query', 'all']:
    #     logger.info("Running query from a string file example")
    #     results = run_cursor_to_pandas_string_example()
    #     if not results.empty:
    #         logger.info(f"Query returned {len(results)} rows")
    
    if args.example in ['query', 'all']:
        logger.info("Running query from a string file example")
        results = run_cursor_to_pandas_file_example()
        if not results.empty:
            logger.info(f"Query returned {len(results)} rows")
    
    logger.info("Examples completed")

if __name__ == "__main__":
    main()