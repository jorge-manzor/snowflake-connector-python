import logging
import pandas as pd
from modules.snowflake_connector import SnowflakeDWH
from config.settings import SNOWFLAKE_CONFIG

logger = logging.getLogger(__name__)

def run_query_to_pandas_string_example():
    """Example of running a query and retrieving results as a DataFrame."""
    try:
        # Initialize Snowflake connector
        snowflake = SnowflakeDWH()
        
        # Example query
        query = f"""
        SELECT 
            fecha_feriado,
            nombre,
            tipo,
            irrenunciable
        FROM 
            {SNOWFLAKE_CONFIG['DATABASE']}.{SNOWFLAKE_CONFIG['SCHEMA']}.{SNOWFLAKE_CONFIG['HOLIDAY_TABLE']}
        WHERE 
            fecha_feriado >= '2025-01-01'
        ORDER BY 
            fecha_feriado
        """
        
        # Execute query and get results as DataFrame
        logger.info("Executing query to retrieve holiday data")
        results_df = snowflake.query_to_pandas(query = query, lowercase_columns = True)
        
        # Example of processing the results
        logger.info(f"Retrieved {len(results_df)} rows from Snowflake")
        
        # Example of filtering and analyzing the data
        irrenunciable_holidays = results_df[results_df['irrenunciable'] == True]
        logger.info(f"Found {len(irrenunciable_holidays)} non-waivable holidays")
        print(results_df)
        
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
    
    logger.info("Running query_to_pandas_string example as standalone script")
    
    # Run the example
    result = run_query_to_pandas_string_example()
    
    # Provide feedback on completion
    if not result.empty:
        logger.info(f"Example completed successfully with {len(result)} rows")
    else:
        logger.error("Example failed or returned no data")

if __name__ == "__main__":
    main()