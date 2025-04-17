# Snowflake Connector

## Snowflake Query Methods Comparison
This document explains the differences between `query_to_pandas` and `cursor_to_pandas` methods and provides guidance on when to use each approach.

### Overview

The Snowflake connector provides two primary methods for executing queries and retrieving results as pandas DataFrames:

1. `query_to_pandas`: Uses SQLAlchemy to execute queries and fetch results
2. `cursor_to_pandas`: Uses Snowflake's native cursor to execute queries and fetch results

Both methods support similar parameters and return pandas DataFrames, but they have different performance characteristics and behaviors.

### Method Signatures
```python
def query_to_pandas(
    query: str = None, 
    filename: str = None, 
    database: Optional[str] = None,
    schema: Optional[str] = None,
    lowercase_columns: bool = True,
    **kwargs
) -> pd.DataFrame

def cursor_to_pandas(
    query: str = None, 
    filename: str = None, 
    database: Optional[str] = None,
    schema: Optional[str] = None,
    lowercase_columns: bool = True,
    **kwargs
) -> pd.DataFrame
```
| **Feature**              | **`query_to_pandas`**                | **`cursor_to_pandas`** |
|--------------------------|--------------------------------------|------------------------------|
| Implementation           | Uses SQLAlchemy with pandas          | Uses native Snowflake cursor |
| Performance for large datasets | May be slower for very large results |	Generally faster for large result sets |
| Memory usage	| Can use more memory	| More memory-efficient |
| Default column case	| Lowercase	| Uppercase (native Snowflake) |
| Connection pooling	| Uses SQLAlchemy engine	| Creates a new connection |
| Error handling	| SQLAlchemy errors	| Native Snowflake errors| 

### When to Use Each Method
Use `query_to_pandas` when:
- Working with smaller to medium-sized result sets (up to a few hundred thousand rows).
- You need SQLAlchemy's additional features.
- You want consistent lowercase column names by default.
- You're using other SQLAlchemy features in your application.
- You need database-agnostic code that might work with other databases.

Use `cursor_to_pandas` when:
- Working with large result sets (hundreds of thousands to millions of rows).
- Performance is critical.
- Memory efficiency is important.
- You need native Snowflake cursor features.
- You prefer native Snowflake column case (uppercase) or explicitly control column case.

### Performance Comparison
In general, `cursor_to_pandas` offers better performance for larger result sets due to:

- Direct use of Snowflake's optimized cursor.
- Reduced overhead from SQLAlchemy's ORM layer.
- More efficient memory management.
- For small to medium queries, the performance difference may be negligible, and other factors like code consistency might be more important.

### Best Practices
1. **For exploratory analysis and smaller datasets:** Use `query_to_pandas` for simplicity.
2. **For production ETL with large datasets:** Use `cursor_to_pandas` for performance.
3. **For consistent column naming:** Set `lowercase_columns` explicitly based on your needs.
4. **For SQL files:** Both methods support SQL file execution with parameter substitution.
5. **For memory-constrained environments:** Prefer `cursor_to_pandas`.