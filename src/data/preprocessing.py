"""
Data preprocessing utilities for JEE College Prediction project.

This module contains functions for cleaning and preprocessing the scraped data.
"""

import pandas as pd
import numpy as np
import pickle
import logging
from typing import Union, List, Dict, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A class for preprocessing JEE admission data.
    
    This class provides methods to clean, transform, and prepare the data
    for machine learning model training.
    """
    
    def __init__(self):
        """Initialize the data preprocessor."""
        self.data = None
        self.processed_data = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a pickle file.
        
        Args:
            file_path (str): Path to the pickle file
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            with open(file_path, 'rb') as f:
                self.data = pickle.load(f)
            logger.info(f"Data loaded successfully from {file_path}")
            return self.data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def clean_rank_data(self, value: Union[str, int, float]) -> Union[int, float]:
        """
        Clean rank data by converting various formats to integer.
        
        Args:
            value: Raw rank value
            
        Returns:
            int or np.nan: Cleaned rank value
        """
        try:
            return int(float(value))
        except ValueError:
            try:
                # Handle cases where rank ends with characters like 'K', 'L', etc.
                if isinstance(value, str) and value[:-1].isdigit():
                    return int(value[:-1])
                else:
                    return np.nan
            except:
                return np.nan
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw data.
        
        Args:
            df (pd.DataFrame): Raw dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        logger.info("Starting data cleaning process...")
        
        # Create a copy to avoid modifying the original data
        cleaned_df = df.copy()
        
        # Remove rows with missing Institute information
        cleaned_df = cleaned_df.dropna(subset=["Institute"])
        logger.info(f"Removed rows with missing Institute data. Remaining rows: {len(cleaned_df)}")
        
        # Fill missing Gender values with "Neutral"
        cleaned_df["Gender"] = cleaned_df["Gender"].fillna("Neutral")
        logger.info("Filled missing Gender values with 'Neutral'")
        
        # Clean rank columns
        if 'Opening Rank' in cleaned_df.columns:
            cleaned_df['Opening Rank'] = cleaned_df['Opening Rank'].apply(self.clean_rank_data)
            logger.info("Cleaned Opening Rank column")
            
        if 'Closing Rank' in cleaned_df.columns:
            cleaned_df['Closing Rank'] = cleaned_df['Closing Rank'].apply(self.clean_rank_data)
            logger.info("Cleaned Closing Rank column")
        
        # Remove rows with invalid rank data
        rank_columns = ['Opening Rank', 'Closing Rank']
        for col in rank_columns:
            if col in cleaned_df.columns:
                initial_count = len(cleaned_df)
                cleaned_df = cleaned_df.dropna(subset=[col])
                logger.info(f"Removed {initial_count - len(cleaned_df)} rows with invalid {col} data")
        
        self.processed_data = cleaned_df
        logger.info("Data cleaning completed successfully")
        return cleaned_df
    
    def save_processed_data(self, df: pd.DataFrame, file_path: str) -> None:
        """
        Save processed data to a pickle file.
        
        Args:
            df (pd.DataFrame): Processed dataframe
            file_path (str): Output file path
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(df, f)
            logger.info(f"Processed data saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of the dataframe.
        
        Args:
            df (pd.DataFrame): Dataframe to summarize
            
        Returns:
            dict: Summary statistics
        """
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'unique_values': {col: df[col].nunique() for col in df.columns}
        }
        
        return summary
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the processed data.
        
        Args:
            df (pd.DataFrame): Dataframe to validate
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        logger.info("Validating processed data...")
        
        # Check for required columns
        required_columns = ['Institute', 'Opening Rank', 'Closing Rank', 'Gender', 'Seat Type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for data types
        if not pd.api.types.is_numeric_dtype(df['Opening Rank']):
            logger.error("Opening Rank column should be numeric")
            return False
            
        if not pd.api.types.is_numeric_dtype(df['Closing Rank']):
            logger.error("Closing Rank column should be numeric")
            return False
        
        # Check for logical consistency
        invalid_ranks = df[df['Opening Rank'] > df['Closing Rank']]
        if len(invalid_ranks) > 0:
            logger.warning(f"Found {len(invalid_ranks)} rows where Opening Rank > Closing Rank")
        
        logger.info("Data validation completed successfully")
        return True


def main():
    """
    Main function to demonstrate data preprocessing.
    """
    preprocessor = DataPreprocessor()
    
    # Example usage
    try:
        # Load raw data
        raw_data = preprocessor.load_data("../../data/raw/data_v1.pkl")
        
        # Clean the data
        cleaned_data = preprocessor.clean_data(raw_data)
        
        # Validate the data
        if preprocessor.validate_data(cleaned_data):
            # Save processed data
            preprocessor.save_processed_data(cleaned_data, "../../data/processed/data_v2.pkl")
            
            # Print summary
            summary = preprocessor.get_data_summary(cleaned_data)
            print("Data Summary:")
            for key, value in summary.items():
                print(f"{key}: {value}")
        else:
            logger.error("Data validation failed")
            
    except Exception as e:
        logger.error(f"Error in main preprocessing: {str(e)}")


if __name__ == "__main__":
    main()
