"""Configuration settings for the application."""

# API settings
API_CONFIG = {
    'HOLIDAY_API_BASE_URL': 'https://api.boostr.cl/holidays',
}

# Snowflake settings
SNOWFLAKE_CONFIG = {
    'DATABASE': 'LATAM_PROD',
    'SCHEMA': 'ANALYST_SANDBOX',
    'HOLIDAY_TABLE': 'TMP_CHILE_HOLIDAY_DAYS_TEST',
}

# Processing settings
PROCESSING_CONFIG = {
    'DEFAULT_START_YEAR': 2025,
    'DEFAULT_END_YEAR': 2026,
    'EXCLUDED_HOLIDAYS': ['Todos los Días Domingos'],
}