import logging
import pandas as pd
from modules.snowflake_connector import SnowflakeDWH
from config.settings import SNOWFLAKE_CONFIG

logger = logging.getLogger(__name__)

def execute_query_string_example():
    """Example of running a query and retrieving results as a DataFrame."""
    try:
        # Initialize Snowflake connector
        snowflake = SnowflakeDWH()
        
        # Example query
        query = 'DROP TABLE latam_prod.source_latam_payment_airflow.cybersource_payment_batch_daily_report;'
        
        # Execute query and get results as DataFrame
        results_df = snowflake.execute_query(query = query)
        
        # Return the DataFrame for further processing
        return results_df
        
    except Exception as e:
        logger.error(f"Error in query example: {e}", exc_info=True)
        return pd.DataFrame()

def main():
    """Main function when running as a script."""
    import sys
    import os
    
    # Get the directory of the current file
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the parent directory (project root)
    project_root = os.path.dirname(current_file_dir)
    
    # Change the current working directory to the project root
    os.chdir(project_root)
    
    # Import and setup logging
    from utils.logging_config import setup_logging
    setup_logging(level=logging.INFO)
    
    logger.info("Running execute_query_string example as standalone script")
    
    # Run the example
    result = execute_query_string_example()
    
    # Provide feedback on completion
    logger.info(f"Query Execution Result:\n{result}")

if __name__ == "__main__":
    main()