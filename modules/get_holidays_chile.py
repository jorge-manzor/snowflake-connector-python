import pandas as pd
import datetime
from typing import List

class DataProcessor:
    """Base class for data processing operations."""
    
    @staticmethod
    def add_metadata(df: pd.DataFrame, metadata_columns: dict = None) -> pd.DataFrame:
        """Add metadata columns to DataFrame."""
        if metadata_columns is None:
            current_time = datetime.datetime.now()
            metadata_columns = {
                'src_created_at': current_time,
                'src_updated_at': current_time
            }
            
        for column, value in metadata_columns.items():
            df[column] = value
            
        return df


class HolidayDataProcessor(DataProcessor):
    """Process and prepare holiday data for Snowflake ingestion."""
    
    @staticmethod
    def add_custom_holidays(df: pd.DataFrame, year: int) -> pd.DataFrame:
        """Add custom holidays that may not be in the API response."""
        if df is None:
            df = pd.DataFrame(columns=['fecha_feriado', 'nombre', 'tipo', 'irrenunciable', 'additional'])
        
        custom_holidays = [
            {
                'fecha_feriado': f'{year}-12-31',
                'nombre': 'Feriado Bancario',
                'tipo': 'Civil',
                'irrenunciable': False,
                'additional': 'Bancario'
            },
            {
                'fecha_feriado': f'{year}-01-01',
                'nombre': 'Año Nuevo',
                'tipo': 'Civil',
                'irrenunciable': True,
                'additional': 'Civil e Irrenunciable'
            }
        ]
        
        for holiday in custom_holidays:
            if not HolidayDataProcessor.date_exists(holiday['fecha_feriado'], df):
                df = pd.concat([df, pd.DataFrame([holiday])], ignore_index=True)
                
        return df
    
    @staticmethod
    def date_exists(date: str, df: pd.DataFrame, date_column: str = 'fecha_feriado') -> bool:
        """Check if a specific date already exists in the DataFrame."""
        if df.empty or date_column not in df.columns:
            return False
        return date in df[date_column].values
    
    @staticmethod
    def filter_holidays(df: pd.DataFrame, exclude_names: List[str] = None) -> pd.DataFrame:
        """Filter out unwanted holidays based on name."""
        if exclude_names is None:
            exclude_names = ['Todos los Días Domingos']
            
        for name in exclude_names:
            df = df[df['nombre'] != name]
            
        return df