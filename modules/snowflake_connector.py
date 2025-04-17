import snowflake.connector
import pandas as pd
import timeit
import yaml
import logging
import uuid
from typing import Dict, Any, Optional, Union, List
from os import environ, path
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL  
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from snowflake.connector.pandas_tools import write_pandas

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('__name__')

class SnowflakeDWH:
    """
    A class to handle Snowflake data warehouse operations including:
    - Authentication
    - Query execution
    - Data transfer between Snowflake and pandas
    - Merge operations for data historization
    """

    def __init__(self, 
                 credentials_file: str = 'credentials/credentials.yml', 
                 credentials_key: str = 'snowflake_dbt',
                 user: Optional[str] = None,
                 account: Optional[str] = None,
                 database: Optional[str] = None,
                 schema: Optional[str] = None,
                 warehouse: Optional[str] = None,
                 role: Optional[str] = None,
                 rsa_key: Optional[str] = None):
        """
        Initialize the Snowflake connection parameters.
        
        Args:
            credentials_file: Path to YAML file containing credentials
            credentials_key: Key in the YAML file for Snowflake credentials
            user, account, database, schema, warehouse, role, rsa_key: 
                Optional parameters to override credentials from file
        """
        # Load credentials from file
        try:
            with open(credentials_file) as f:
                creds = yaml.safe_load(f)
                credentials = creds.get(credentials_key, {})
        except Exception as e:
            logger.warning(f"Could not load credentials from {credentials_file}: {e}")
            credentials = {}
        
        # Use provided parameters or fall back to credentials file
        self.user = user or credentials.get('user')
        self.account = account or credentials.get('account')
        self.database = database or credentials.get('database')
        self.schema = schema or credentials.get('schema')
        self.warehouse = warehouse or credentials.get('warehouse')
        self.role = role or credentials.get('role')
        rsa_key_str = rsa_key or credentials.get('rsa_key')
        
        # Check for required parameters
        required_params = ['user', 'account', 'warehouse', 'role']
        missing_params = [param for param in required_params if not getattr(self, param)]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")
        
        # Database and schema are optional but useful to log a warning
        if not self.database:
            logger.warning("No database specified, will use user's default database")
        if not self.schema:
            logger.warning("No schema specified, will use user's default schema")
        
        if not rsa_key_str:
            raise ValueError("RSA key is required")
            
        # Process RSA key
        self.private_key_bytes = self._process_rsa_key(rsa_key_str)
        
        # Initialize connection objects as None
        self.engine = None
        self.connection = None
        self.cur = None
        
    def _process_rsa_key(self, rsa_key_str: str) -> bytes:
        """
        Process RSA key string into bytes format required by Snowflake.
        
        Args:
            rsa_key_str: RSA private key in PEM format
            
        Returns:
            Private key in DER format
        """
        try:
            p_key = serialization.load_pem_private_key(
                rsa_key_str.encode('utf-8'),
                password=None,
                backend=default_backend()
            )
                
            return p_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        except Exception as e:
            raise ValueError(f"Invalid RSA key: {e}")

    def get_engine(self, database: Optional[str] = None, schema: Optional[str] = None):
        """
        Create and return a SQLAlchemy engine for Snowflake.
        
        Args:
            database: Optional database override
            schema: Optional schema override
            
        Returns:
            SQLAlchemy engine object
        """
        # Use provided values or instance defaults
        db = database or self.database
        schema_name = schema or self.schema
        
        # Build connection parameters
        connect_args = {'private_key': self.private_key_bytes}
        
        # URI for connection to DWH
        db_uri_params = {
            'account': self.account,
            'user': self.user,
            'warehouse': self.warehouse,
            'role': self.role
        }
        
        # Add optional parameters if provided
        if db:
            db_uri_params['database'] = db
        if schema_name:
            db_uri_params['schema'] = schema_name
            
        db_uri = URL(**db_uri_params)

        # Create SQLAlchemy engine        
        self.engine = create_engine(db_uri, connect_args=connect_args)
        return self.engine

    def get_connection(self, database: Optional[str] = None, schema: Optional[str] = None):
        """
        Create and return a native Snowflake connection.
        
        Args:
            database: Optional database override
            schema: Optional schema override
            
        Returns:
            Snowflake connection object
        """
        # Use provided values or instance defaults
        db = database or self.database
        schema_name = schema or self.schema
        
        # Build connection parameters
        connect_params = {
            'user': self.user,
            'private_key': self.private_key_bytes,
            'account': self.account,
            'warehouse': self.warehouse,
            'role': self.role
        }
        
        # Add optional parameters if provided
        if db:
            connect_params['database'] = db
        if schema_name:
            connect_params['schema'] = schema_name
            
        self.connection = snowflake.connector.connect(**connect_params)
        return self.connection

    def parse_query(self, filename: str, **kwargs) -> str:
        """
        Read a SQL query from file and replace parameters.
        
        Args:
            filename: Path to SQL file
            **kwargs: Parameters to replace in the query
            
        Returns:
            Processed SQL query
        """
        try:
            # Read query from file
            with open(filename, "r") as f:
                query = f.read()
                
            # Replace relevant parameters
            for key, value in kwargs.items():
                if 'list' in key:
                    # More robust list parameter handling
                    placeholder = f"{{{key}}}"
                    if placeholder in query:
                        query = query.replace(placeholder, value)
                else:
                    query = query.replace(key, str(value))
                    
            return query
        except Exception as e:
            logger.error(f"Error parsing query from {filename}: {e}")
            raise

    def run_query(self, 
                 query: str, 
                 params: Optional[Dict[str, Any]] = None,
                 database: Optional[str] = None,
                 schema: Optional[str] = None) -> Any:
        """
        Execute a SQL query and return the result.
        
        Args:
            query: SQL query to execute
            params: Optional parameters for parameterized queries
            database: Optional database override
            schema: Optional schema override
            
        Returns:
            Query result
        """
        start_time = timeit.default_timer()
        try:
            with self.get_engine(database=database, schema=schema).connect() as conn:
                if params:
                    result = conn.execute(query, params)
                else:
                    result = conn.execute(query)
                logger.info(f"Query executed in {timeit.default_timer() - start_time:.2f} seconds")
                return result
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
        finally:
            if self.engine:
                self.engine.dispose()
    
    def query_to_pandas(self, 
                       query: str = None, 
                       filename: str = None, 
                       database: Optional[str] = None,
                       schema: Optional[str] = None,
                       lowercase_columns: bool = True,
                       **kwargs) -> pd.DataFrame:
        """
        Execute a query and return results as a pandas DataFrame.
        
        Args:
            query: SQL query string (optional if filename is provided)
            filename: Path to SQL file (optional if query is provided)
            database: Optional database override
            schema: Optional schema override
            lowercase_columns: Whether to convert column names to lowercase (default: True)
            **kwargs: Parameters to replace in the query if using a file
            
        Returns:
            pandas DataFrame with query results
        """
        start_time = timeit.default_timer()
        
        if not query and not filename:
            raise ValueError("Either query or filename must be provided")
            
        try:
            engine = self.get_engine(database=database, schema=schema)
            
            if filename:
                query = self.parse_query(filename=filename, **kwargs)
                
            # Download database from query
            df = pd.read_sql_query(sql=query, con=engine)
            
            # Handle column case based on parameter
            if not lowercase_columns:
                # Convert columns to uppercase to match Snowflake's default
                df.columns = df.columns.str.upper()
            
            execution_time = timeit.default_timer() - start_time
            logger.info(f'Snowflake query execution time: {execution_time:.2f} seconds')
            return df
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
        finally:
            if self.engine:
                self.engine.dispose()

    def cursor_to_pandas(self, 
                        query: str = None, 
                        filename: str = None, 
                        database: Optional[str] = None,
                        schema: Optional[str] = None,
                        lowercase_columns: bool = True,
                        **kwargs) -> pd.DataFrame:
        """
        Execute a query using native cursor and return results as a pandas DataFrame.
        Can be more efficient for large result sets.
        
        Args:
            query: SQL query string (optional if filename is provided)
            filename: Path to SQL file (optional if query is provided)
            database: Optional database override
            schema: Optional schema override
            lowercase_columns: Whether to convert column names to lowercase (default: True)
            **kwargs: Parameters to replace in the query if using a file
            
        Returns:
            pandas DataFrame with query results
        """
        start_time = timeit.default_timer()
        
        if not query and not filename:
            raise ValueError("Either query or filename must be provided")
            
        conn = None
        cursor = None
        
        try:
            conn = self.get_connection(database=database, schema=schema)
            cursor = conn.cursor()

            if filename:
                query = self.parse_query(filename=filename, **kwargs)
                
            cursor.execute(query)
            df = cursor.fetch_pandas_all()
            
            # Handle column case based on parameter
            if lowercase_columns:
                # Convert columns to uppercase to match Snowflake's default
                df.columns = df.columns.str.lower()
            
            execution_time = timeit.default_timer() - start_time
            logger.info(f'Snowflake query execution time: {execution_time:.2f} seconds')
            return df
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def pandas_to_dwh(self, 
                     dataframe: pd.DataFrame, 
                     table_name: str, 
                     schema_name: Optional[str] = None,
                     database_name: Optional[str] = None,
                     if_exists: str = 'fail', 
                     dtype: Optional[Dict] = None,
                     chunk_size: int = 10000) -> bool:
        """
        Upload a pandas DataFrame to Snowflake.
        
        Args:
            dataframe: pandas DataFrame to upload
            table_name: Target table name
            schema_name: Target schema (defaults to connection schema if None)
            database_name: Target database (defaults to connection database if None)
            if_exists: Action if table exists ('fail', 'replace', 'append')
            dtype: Column data types
            chunk_size: Number of rows per batch
            
        Returns:
            True if successful
        """
        start_time = timeit.default_timer()
        schema_name = schema_name or self.schema
        database_name = database_name or self.database
        
        try:
            engine = self.get_engine(database=database_name, schema=schema_name)
            
            dataframe.to_sql(
                name=table_name, 
                con=engine,
                schema=schema_name,
                if_exists=if_exists,
                dtype=dtype,
                index=False,
                chunksize=chunk_size
            )
            
            execution_time = timeit.default_timer() - start_time
            logger.info(f'Data upload completed in {execution_time:.2f} seconds')
            return True
            
        except Exception as e:
            logger.error(f"Error uploading data: {e}")
            raise
        finally:
            if self.engine:
                self.engine.dispose()
                
    def merge_dataframe(self,
                    dataframe: pd.DataFrame,
                    target_table: str,
                    temp_table: Optional[str] = None,
                    schema_name: Optional[str] = None,
                    database_name: Optional[str] = None,
                    key_columns: List[str] = None,
                    update_columns: Optional[List[str]] = None,
                    chunk_size: int = 10000) -> Dict[str, Any]:
        """
        Perform a merge operation (upsert) from a pandas DataFrame to a Snowflake table.
        """
        if not key_columns:
            raise ValueError("key_columns must be specified for merge operation")
            
        schema_name = schema_name or self.schema
        database_name = database_name or self.database
        
        # Generate temp table name if not provided
        if not temp_table:
            temp_table = f"TEMP_{target_table}_{str(uuid.uuid4()).replace('-', '')[:8]}"
        
        # Build unquoted table names - Snowflake will handle the quoting properly
        if schema_name and database_name:
            qualified_target = f"{database_name}.{schema_name}.{target_table}"
            qualified_temp = f"{database_name}.{schema_name}.{temp_table}"
        elif schema_name:
            qualified_target = f"{schema_name}.{target_table}"
            qualified_temp = f"{schema_name}.{temp_table}"
        else:
            qualified_target = target_table
            qualified_temp = temp_table
        
        conn = None
        cursor = None
        
        try:
            # Create a single connection to use throughout the process
            conn = self.get_connection(database=database_name, schema=schema_name)
            cursor = conn.cursor()
            
            # Step 1: Create a temporary table with the same structure as the target
            logger.info(f"Creating temporary table {qualified_temp}")
            cursor.execute(f"CREATE TEMPORARY TABLE {qualified_temp} LIKE {qualified_target}")
            
            # Step 2: Load data using direct INSERT VALUES approach
            logger.info(f"Loading data into temporary table {temp_table}")
            
            # Make a copy of the dataframe to avoid modifying the original
            df_copy = dataframe.copy()
            
            # Convert datetime columns to ISO format strings that Snowflake can parse
            for col in df_copy.select_dtypes(include=['datetime64']).columns:
                df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
            
            # Build INSERT statement with VALUES clause
            columns = list(df_copy.columns)
            column_list = ", ".join(columns)
            
            # Generate VALUES clause for each row
            values_list = []
            for _, row in df_copy.iterrows():
                values = []
                for col in columns:
                    if pd.isna(row[col]):
                        values.append("NULL")
                    elif isinstance(row[col], (int, float)):
                        values.append(str(row[col]))
                    else:
                        # Escape single quotes in string values
                        val = str(row[col])
                        escaped_val = val.replace("'", "''")
                        values.append(f"'{escaped_val}'")
                values_list.append(f"({', '.join(values)})")
            
            # Execute INSERT in batches to avoid statement size limits
            batch_size = 1000  # Smaller batch size for safety
            for i in range(0, len(values_list), batch_size):
                batch_values = values_list[i:i+batch_size]
                insert_sql = f"INSERT INTO {qualified_temp} ({column_list}) VALUES {', '.join(batch_values)}"
                cursor.execute(insert_sql)
            
            # Verify data was loaded
            logger.info(f"Verifying data was loaded into {qualified_temp}")
            cursor.execute(f"SELECT COUNT(*) FROM {qualified_temp}")
            count_result = cursor.fetchone()
            row_count = count_result[0] if count_result else 0
            logger.info(f"Rows in temporary table: {row_count}")
            
            if row_count == 0:
                raise ValueError(f"No data was loaded into temporary table {qualified_temp}")
            
            # Step 3: Determine columns to update if not specified
            if update_columns is None:
                all_columns = list(dataframe.columns)
                update_columns = [col for col in all_columns if col not in key_columns]
            
            # Step 4: Build and execute MERGE statement
            merge_sql = f"""
            MERGE INTO {qualified_target} target
            USING {qualified_temp} source
            ON {' AND '.join([f'target.{col} = source.{col}' for col in key_columns])}
            WHEN MATCHED THEN UPDATE SET
                {', '.join([f'{col} = source.{col}' for col in update_columns])}
            WHEN NOT MATCHED THEN INSERT
                ({', '.join(dataframe.columns)})
            VALUES
                ({', '.join([f'source.{col}' for col in dataframe.columns])})
            """
            
            logger.info("Executing MERGE operation")
            cursor.execute(merge_sql)
            
            # Get merge operation stats
            result_count = cursor.fetchone()[0] if cursor.description else cursor.rowcount
            logger.info(f"MERGE affected {result_count} rows")
            
            # Step 5: Clean up temporary table
            logger.info(f"Dropping temporary table {qualified_temp}")
            cursor.execute(f"DROP TABLE IF EXISTS {qualified_temp}")
            
            return {
                "success": True,
                "affected_rows": result_count,
                "source_rows": len(dataframe)
            }
            
        except Exception as e:
            logger.error(f"Error during merge operation: {e}")
            # Try to clean up the temporary resources if they exist
            try:
                if cursor:
                    if 'qualified_temp' in locals():
                        cursor.execute(f"DROP TABLE IF EXISTS {qualified_temp}")
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up resources: {cleanup_error}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def execute_custom_merge(self, 
                            source_data: Union[pd.DataFrame, str],
                            target_table: str,
                            merge_sql: Optional[str] = None,
                            merge_sql_file: Optional[str] = None,
                            schema_name: Optional[str] = None,
                            database_name: Optional[str] = None,
                            temp_table: Optional[str] = None,
                            chunk_size: int = 10000,
                            **kwargs) -> Dict[str, Any]:
        """
        Execute a custom merge operation using provided SQL or SQL file.
        Allows for complex merge logic beyond simple upserts.
        
        Args:
            source_data: Either a DataFrame to load or a table name to use as source
            target_table: Target table for the merge
            merge_sql: Custom merge SQL statement with {placeholders}
            merge_sql_file: File containing merge SQL
            schema_name: Schema name (defaults to connection schema)
            database_name: Database name (defaults to connection database)
            temp_table: Temporary table name if loading DataFrame
            chunk_size: Batch size for loading data
            **kwargs: Parameters to substitute in the SQL
            
        Returns:
            Dictionary with operation results
        """
        schema_name = schema_name or self.schema
        database_name = database_name or self.database
        
        # Fully qualify target table name
        qualified_target = f"{target_table}"
        if schema_name:
            qualified_target = f"{schema_name}.{qualified_target}"
        if database_name:
            qualified_target = f"{database_name}.{qualified_target}"
        
        # Get the merge SQL
        if merge_sql_file:
            merge_sql = self.parse_query(filename=merge_sql_file, **kwargs)
        elif not merge_sql:
            raise ValueError("Either merge_sql or merge_sql_file must be provided")
        
        conn = None
        cursor = None
        
        try:
            # Create a single connection to use throughout the process
            conn = self.get_connection(database=database_name, schema=schema_name)
            cursor = conn.cursor()
            
            # If source_data is a DataFrame, load it to a temp table
            if isinstance(source_data, pd.DataFrame):
                if not temp_table:
                    temp_table = f"TEMP_MERGE_{str(uuid.uuid4()).replace('-', '')[:8]}"
                    
                qualified_source = f"{temp_table}"
                if schema_name:
                    qualified_source = f"{schema_name}.{qualified_source}"
                if database_name:
                    qualified_source = f"{database_name}.{qualified_source}"
                
                # Create a temporary table like the target
                logger.info(f"Creating temporary table {qualified_source}")
                cursor.execute(f"CREATE TEMPORARY TABLE {qualified_source} LIKE {qualified_target}")
                
                # Load data using direct INSERT VALUES approach
                logger.info(f"Loading data into temporary table {temp_table}")
                
                # Make a copy of the dataframe to avoid modifying the original
                df_copy = source_data.copy()
                
                # Convert datetime columns to ISO format strings that Snowflake can parse
                for col in df_copy.select_dtypes(include=['datetime64']).columns:
                    df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
                
                # Build INSERT statement with VALUES clause
                columns = list(df_copy.columns)
                column_list = ", ".join(columns)
                
                # Generate VALUES clause for each row
                values_list = []
                for _, row in df_copy.iterrows():
                    values = []
                    for col in columns:
                        if pd.isna(row[col]):
                            values.append("NULL")
                        elif isinstance(row[col], (int, float)):
                            values.append(str(row[col]))
                        else:
                            # Escape single quotes in string values
                            val = str(row[col])
                            escaped_val = val.replace("'", "''")
                            values.append(f"'{escaped_val}'")
                    values_list.append(f"({', '.join(values)})")
                
                # Execute INSERT in batches to avoid statement size limits
                batch_size = 1000  # Smaller batch size for safety
                for i in range(0, len(values_list), batch_size):
                    batch_values = values_list[i:i+batch_size]
                    insert_sql = f"INSERT INTO {qualified_source} ({column_list}) VALUES {', '.join(batch_values)}"
                    cursor.execute(insert_sql)
                
                # Verify data was loaded
                logger.info(f"Verifying data was loaded into {qualified_source}")
                cursor.execute(f"SELECT COUNT(*) FROM {qualified_source}")
                count_result = cursor.fetchone()
                row_count = count_result[0] if count_result else 0
                logger.info(f"Rows in temporary table: {row_count}")
                
                if row_count == 0:
                    raise ValueError(f"No data was loaded into temporary table {qualified_source}")
                
            else:
                # Use the provided table name
                qualified_source = f"{source_data}"
                if '.' not in source_data:
                    if schema_name:
                        qualified_source = f"{schema_name}.{qualified_source}"
                    if database_name:
                        qualified_source = f"{database_name}.{qualified_source}"
            
            # Replace placeholders in the SQL
            final_sql = merge_sql.format(
                source_table=qualified_source,
                target_table=qualified_target,
                **kwargs
            )
            
            # Execute the merge
            logger.info("Executing custom merge operation")
            logger.info(f"Merge SQL: {final_sql}")
            cursor.execute(final_sql)
            
            # Get operation stats
            result_count = cursor.fetchone()[0] if cursor.description else cursor.rowcount
            logger.info(f"Merge affected {result_count} rows")
            
            # Clean up if we created a temp table
            if isinstance(source_data, pd.DataFrame) and temp_table:
                logger.info(f"Dropping temporary table {qualified_source}")
                cursor.execute(f"DROP TABLE IF EXISTS {qualified_source}")
            
            return {
                "success": True,
                "affected_rows": result_count,
                "source_rows": len(source_data) if isinstance(source_data, pd.DataFrame) else None
            }
            
        except Exception as e:
            logger.error(f"Error during custom merge operation: {e}")
            # Try to clean up temporary resources
            try:
                if cursor and isinstance(source_data, pd.DataFrame) and 'qualified_source' in locals():
                    cursor.execute(f"DROP TABLE IF EXISTS {qualified_source}")
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up resources: {cleanup_error}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()


    def historize_data(self,
                    dataframe: pd.DataFrame,
                    target_table: str,
                    schema_name: Optional[str] = None,
                    database_name: Optional[str] = None,
                    key_columns: List[str] = None,
                    effective_from_col: str = "effective_from",
                    effective_to_col: str = "effective_to",
                    current_indicator_col: Optional[str] = "is_current",
                    audit_columns: Optional[Dict[str, str]] = None,
                    chunk_size: int = 10000) -> Dict[str, Any]:
        """
        Implement a Type 2 Slowly Changing Dimension (SCD) pattern to maintain full history.
        
        Args:
            dataframe: Source pandas DataFrame with new/updated records
            target_table: Target history table
            schema_name: Schema name (defaults to connection schema)
            database_name: Database name (defaults to connection database)
            key_columns: List of business key columns that identify unique entities
            effective_from_col: Column name for the start date of a record's validity
            effective_to_col: Column name for the end date of a record's validity
            current_indicator_col: Column name for the flag indicating current record
            audit_columns: Dictionary mapping of audit columns to add/update
                        e.g. {"created_by": "SYSTEM", "created_at": "CURRENT_TIMESTAMP()"}
            chunk_size: Batch size for loading data
            
        Returns:
            Dictionary with operation results
        """
        if not key_columns:
            raise ValueError("key_columns must be specified for historization")
        
        schema_name = schema_name or self.schema
        database_name = database_name or self.database
        
        # Generate temp table name
        temp_table = f"TEMP_HIST_{str(uuid.uuid4()).replace('-', '')[:8]}"
        
        # Ensure the dataframe has the required SCD columns
        if effective_from_col not in dataframe.columns:
            raise ValueError(f"DataFrame must contain {effective_from_col} column")
        
        # Add effective_to and current indicator if they don't exist
        if effective_to_col not in dataframe.columns:
            dataframe[effective_to_col] = None  # Will be filled with NULL/None in Snowflake
        
        if current_indicator_col and current_indicator_col not in dataframe.columns:
            dataframe[current_indicator_col] = True
        
        # Fully qualify table names
        qualified_target = f"{target_table}"
        qualified_temp = f"{temp_table}"
        
        if schema_name:
            qualified_target = f"{schema_name}.{qualified_target}"
            qualified_temp = f"{schema_name}.{qualified_temp}"
            
        if database_name:
            qualified_target = f"{database_name}.{qualified_target}"
            qualified_temp = f"{database_name}.{qualified_temp}"
        
        conn = None
        cursor = None
        
        try:
            # Create a single connection to use throughout the process
            conn = self.get_connection(database=database_name, schema=schema_name)
            cursor = conn.cursor()
            
            # Step 1: Create a temporary table with the same structure as the target
            logger.info(f"Creating temporary table {qualified_temp}")
            cursor.execute(f"CREATE TEMPORARY TABLE {qualified_temp} LIKE {qualified_target}")
            
            # Step 2: Load data using direct INSERT VALUES approach
            logger.info(f"Loading data into temporary table {temp_table}")
            
            # Make a copy of the dataframe to avoid modifying the original
            df_copy = dataframe.copy()
            
            # Convert datetime columns to ISO format strings that Snowflake can parse
            for col in df_copy.select_dtypes(include=['datetime64']).columns:
                df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
            
            # Build INSERT statement with VALUES clause
            columns = list(df_copy.columns)
            column_list = ", ".join(columns)
            
            # Generate VALUES clause for each row
            values_list = []
            for _, row in df_copy.iterrows():
                values = []
                for col in columns:
                    if pd.isna(row[col]):
                        values.append("NULL")
                    elif isinstance(row[col], (int, float)):
                        values.append(str(row[col]))
                    else:
                        # Escape single quotes in string values
                        val = str(row[col])
                        escaped_val = val.replace("'", "''")
                        values.append(f"'{escaped_val}'")
                values_list.append(f"({', '.join(values)})")
            
            # Execute INSERT in batches to avoid statement size limits
            batch_size = 1000  # Smaller batch size for safety
            for i in range(0, len(values_list), batch_size):
                batch_values = values_list[i:i+batch_size]
                insert_sql = f"INSERT INTO {qualified_temp} ({column_list}) VALUES {', '.join(batch_values)}"
                cursor.execute(insert_sql)
            
            # Verify data was loaded
            logger.info(f"Verifying data was loaded into {qualified_temp}")
            cursor.execute(f"SELECT COUNT(*) FROM {qualified_temp}")
            count_result = cursor.fetchone()
            row_count = count_result[0] if count_result else 0
            logger.info(f"Rows in temporary table: {row_count}")
            
            if row_count == 0:
                raise ValueError(f"No data was loaded into temporary table {qualified_temp}")
            
            # Step 3: Build and execute the SCD Type 2 merge
            # Build the key matching condition
            key_match = ' AND '.join([f"target.{col} = source.{col}" for col in key_columns])
            
            # Build the change detection condition (all columns except keys and SCD columns)
            scd_cols = [effective_from_col, effective_to_col]
            if current_indicator_col:
                scd_cols.append(current_indicator_col)
                
            # Add any audit columns to the exclusion list
            audit_cols = list(audit_columns.keys()) if audit_columns else []
            
            # Determine columns to check for changes
            all_columns = list(dataframe.columns)
            change_cols = [col for col in all_columns if col not in key_columns + scd_cols + audit_cols]
            
            # Build change detection condition
            if change_cols:
                change_condition = ' OR '.join([f"target.{col} <> source.{col} OR (target.{col} IS NULL AND source.{col} IS NOT NULL) OR (target.{col} IS NOT NULL AND source.{col} IS NULL)" for col in change_cols])
            else:
                # If no change columns, just use a simple condition that's always false
                change_condition = "FALSE"
            
            # Prepare audit column assignments
            audit_updates = ""
            if audit_columns:
                audit_updates = ", " + ", ".join([f"{col} = {val}" for col, val in audit_columns.items()])
            
            # The SCD Type 2 merge SQL
            historize_sql = f"""
            -- Step 1: Update existing current records that have changes (expire them)
            UPDATE {qualified_target} target
            SET 
                {effective_to_col} = source.{effective_from_col},
                {current_indicator_col if current_indicator_col else "'dummy'"} = FALSE{audit_updates}
            FROM {qualified_temp} source
            WHERE {key_match}
            AND target.{current_indicator_col if current_indicator_col else "'dummy'"} = TRUE
            AND ({change_condition});
            
            -- Step 2: Insert new versions of changed records
            INSERT INTO {qualified_target}
            SELECT source.*
            FROM {qualified_temp} source
            JOIN {qualified_target} target
            ON {key_match}
            WHERE target.{current_indicator_col if current_indicator_col else "'dummy'"} = FALSE
            AND target.{effective_to_col} = source.{effective_from_col};
            
            -- Step 3: Insert completely new records
            INSERT INTO {qualified_target}
            SELECT source.*
            FROM {qualified_temp} source
            WHERE NOT EXISTS (
                SELECT 1 
                FROM {qualified_target} target 
                WHERE {key_match}
            );
            """
            
            logger.info("Executing SCD Type 2 historization")
            cursor.execute(historize_sql)
            
            # Get operation stats - try to get detailed counts if available
            try:
                # First try to get counts from the last statement
                insert_count = cursor.rowcount
                
                # Execute additional queries to get more detailed stats
                cursor.execute(f"""
                SELECT 
                    SUM(CASE WHEN {effective_to_col} IS NOT NULL AND {current_indicator_col if current_indicator_col else "'dummy'"} = FALSE THEN 1 ELSE 0 END) as expired_records,
                    SUM(CASE WHEN {effective_to_col} IS NULL AND {current_indicator_col if current_indicator_col else "'dummy'"} = TRUE THEN 1 ELSE 0 END) as current_records
                FROM {qualified_target}
                WHERE {' OR '.join([f"{col} IN (SELECT {col} FROM {qualified_temp})" for col in key_columns])}
                """)
                
                stats = cursor.fetchone()
                expired_count = stats[0] if stats else 0
                current_count = stats[1] if stats else 0
                
                detailed_stats = {
                    "expired_records": int(expired_count) if expired_count is not None else 0,
                    "current_records": int(current_count) if current_count is not None else 0,
                    "total_affected": int(insert_count) if insert_count is not None else 0
                }
            except Exception as e:
                # If detailed stats fail, just use the basic count
                logger.warning(f"Could not get detailed historization stats: {e}")
                detailed_stats = {"total_affected": cursor.rowcount}
            
            # Step 4: Clean up temporary table
            logger.info(f"Dropping temporary table {qualified_temp}")
            cursor.execute(f"DROP TABLE IF EXISTS {qualified_temp}")
            
            return {
                "success": True,
                "source_rows": len(dataframe),
                "stats": detailed_stats,
                "historization_type": "SCD Type 2",
                "target_table": qualified_target
            }
            
        except Exception as e:
            logger.error(f"Error during historization: {e}")
            # Try to clean up temp table if it exists
            try:
                if cursor and 'qualified_temp' in locals():
                    cursor.execute(f"DROP TABLE IF EXISTS {qualified_temp}")
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up resources: {cleanup_error}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()