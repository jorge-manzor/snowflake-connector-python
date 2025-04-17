import logging
import pandas as pd
from modules.snowflake_connector import SnowflakeDWH
from config.settings import SNOWFLAKE_CONFIG

logger = logging.getLogger(__name__)

def run_cursor_to_pandas_string_example():
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
        results_df = snowflake.cursor_to_pandas(query=query, lowercase_columns=True)
        
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

if __name__ == "__main__":
    # Run the example
    run_cursor_to_pandas_string_example()