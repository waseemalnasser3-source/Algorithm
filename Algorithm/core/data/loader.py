"""
Data Ingestion Module
Multi-format data loading and initial parsing system
Developed by: Nour Al-Din (Data Pipeline Engineer)
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DataLoader:
    """
    DataLoader class for loading datasets from various file formats
    Supports CSV and Excel files
    """
    
    def __init__(self):
        """Initialize the DataLoader"""
        self.supported_formats = ['.csv', '.xlsx', '.xls']
    
    def load_csv(self, filepath, encoding='utf-8'):
        """
        Load data from CSV file
        
        Args:
            filepath: Path to CSV file
            encoding: File encoding (default: utf-8)
            
        Returns:
            pandas DataFrame containing the loaded data
        """
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            print(f"Successfully loaded CSV: {filepath}")
            print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"Error loading CSV file: {str(e)}")
            raise
    
    def load_excel(self, filepath, sheet_name=0):
        """
        Load data from Excel file
        
        Args:
            filepath: Path to Excel file
            sheet_name: Sheet name or index (default: 0)
            
        Returns:
            pandas DataFrame containing the loaded data
        """
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            print(f"Successfully loaded Excel: {filepath}")
            print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"Error loading Excel file: {str(e)}")
            raise
    
    def load_file(self, filepath):
        """
        Automatically detect file type and load accordingly
        
        Args:
            filepath: Path to data file
            
        Returns:
            pandas DataFrame containing the loaded data
        """
        path = Path(filepath)
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Get file extension
        extension = path.suffix.lower()
        
        # Check if format is supported
        if extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {extension}")
        
        # Load based on extension
        if extension == '.csv':
            return self.load_csv(filepath)
        elif extension in ['.xlsx', '.xls']:
            return self.load_excel(filepath)
    
    def get_dataset_info(self, df):
        """
        Get basic information about the dataset
        
        Args:
            df: pandas DataFrame
            
        Returns:
            dict containing dataset information
        """
        info = {
            'rows': len(df),
            'columns': len(df.columns),
            'features': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        return info
    
    def preview_data(self, df, n_rows=5):
        """
        Display preview of the dataset
        
        Args:
            df: pandas DataFrame
            n_rows: Number of rows to display (default: 5)
        """
        print("\n" + "="*60)
        print("DATASET PREVIEW")
        print("="*60)
        print(f"\nFirst {n_rows} rows:")
        print(df.head(n_rows))
        
        print(f"\n\nDataset Info:")
        print(f"Total Rows: {len(df)}")
        print(f"Total Columns: {len(df.columns)}")
        print(f"\nColumn Names: {df.columns.tolist()}")
        print(f"\nData Types:\n{df.dtypes}")
        print(f"\nMissing Values:\n{df.isnull().sum()}")
        print("="*60)


# Example usage
if __name__ == '__main__':
    # Test the DataLoader
    loader = DataLoader()
    
    # Load iris dataset
    try:
        df = loader.load_file('sample_datasets/iris.csv')
        loader.preview_data(df)
        
        # Get dataset info
        info = loader.get_dataset_info(df)
        print(f"\nDataset has {info['rows']} rows and {info['columns']} columns")
        
    except Exception as e:
        print(f"Error: {str(e)}")

