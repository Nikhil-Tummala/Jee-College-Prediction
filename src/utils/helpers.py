"""
Utility functions for the JEE College Prediction project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import logging
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level (str): Logging level
        log_file (str, optional): Log file path
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if log_file:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format
        )


def create_directory_structure(base_path: str) -> None:
    """
    Create the standard directory structure for the project.
    
    Args:
        base_path (str): Base path for the project
    """
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "notebooks",
        "src/data",
        "src/models",
        "src/utils",
        "tests",
        "docs",
        "config"
    ]
    
    for directory in directories:
        full_path = os.path.join(base_path, directory)
        os.makedirs(full_path, exist_ok=True)
        logger.info(f"Created directory: {full_path}")


def validate_data_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that dataframe contains required columns.
    
    Args:
        df (pd.DataFrame): Dataframe to validate
        required_columns (List[str]): List of required column names
        
    Returns:
        bool: True if all required columns are present
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    logger.info("All required columns are present")
    return True


def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        Dict[str, Any]: Data information
    """
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'null_counts': df.isnull().sum().to_dict(),
        'null_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
        'unique_counts': df.nunique().to_dict(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    # Add statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        info['numeric_statistics'] = df[numeric_cols].describe().to_dict()
    
    # Add value counts for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        info['categorical_value_counts'] = {}
        for col in categorical_cols:
            if df[col].nunique() <= 20:  # Only for columns with reasonable number of unique values
                info['categorical_value_counts'][col] = df[col].value_counts().to_dict()
    
    return info


def plot_data_distribution(df: pd.DataFrame, columns: List[str], 
                          figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot distribution of specified columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (List[str]): Columns to plot
        figsize (Tuple[int, int]): Figure size
    """
    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3  # 3 plots per row
    
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(columns):
        row = i // 3
        col_idx = i % 3
        
        if df[col].dtype in ['int64', 'float64']:
            # Numeric column - histogram
            axes[row, col_idx].hist(df[col].dropna(), bins=30, alpha=0.7)
            axes[row, col_idx].set_title(f'Distribution of {col}')
            axes[row, col_idx].set_xlabel(col)
            axes[row, col_idx].set_ylabel('Frequency')
        else:
            # Categorical column - bar plot
            value_counts = df[col].value_counts().head(10)  # Top 10 values
            axes[row, col_idx].bar(range(len(value_counts)), value_counts.values)
            axes[row, col_idx].set_title(f'Distribution of {col}')
            axes[row, col_idx].set_xlabel(col)
            axes[row, col_idx].set_ylabel('Count')
            axes[row, col_idx].set_xticks(range(len(value_counts)))
            axes[row, col_idx].set_xticklabels(value_counts.index, rotation=45)
    
    # Hide empty subplots
    for i in range(n_cols, n_rows * 3):
        row = i // 3
        col_idx = i % 3
        axes[row, col_idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def create_correlation_matrix(df: pd.DataFrame, numeric_only: bool = True) -> None:
    """
    Create and display correlation matrix.
    
    Args:
        df (pd.DataFrame): Input dataframe
        numeric_only (bool): Whether to include only numeric columns
    """
    if numeric_only:
        corr_df = df.select_dtypes(include=[np.number])
    else:
        corr_df = df
    
    if len(corr_df.columns) < 2:
        logger.warning("Not enough numeric columns for correlation matrix")
        return
    
    correlation_matrix = corr_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()


def remove_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.DataFrame:
    """
    Remove outliers from a specific column.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to remove outliers from
        method (str): Method to use ('iqr' or 'zscore')
        
    Returns:
        pd.DataFrame: Dataframe with outliers removed
    """
    if column not in df.columns:
        logger.error(f"Column '{column}' not found in dataframe")
        return df
    
    original_length = len(df)
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_cleaned = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        df_cleaned = df[z_scores < 3]
    
    else:
        logger.error(f"Unknown method: {method}")
        return df
    
    removed_count = original_length - len(df_cleaned)
    logger.info(f"Removed {removed_count} outliers from column '{column}' using {method} method")
    
    return df_cleaned


def export_to_csv(df: pd.DataFrame, file_path: str, index: bool = False) -> None:
    """
    Export dataframe to CSV file.
    
    Args:
        df (pd.DataFrame): Dataframe to export
        file_path (str): Output file path
        index (bool): Whether to include index
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=index)
        logger.info(f"Data exported to {file_path}")
    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        raise


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    import json
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in configuration file: {config_path}")
        raise


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        config_path (str): Output file path
    """
    import json
    
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        raise


class Timer:
    """
    Simple timer context manager for measuring execution time.
    """
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting {self.name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"{self.name} completed in {duration:.2f} seconds")


# Example usage
if __name__ == "__main__":
    # Create a sample dataframe for testing
    sample_data = pd.DataFrame({
        'rank': np.random.randint(1, 10000, 1000),
        'gender': np.random.choice(['Male', 'Female'], 1000),
        'seat_type': np.random.choice(['Open', 'SC', 'ST', 'OBC'], 1000),
        'score': np.random.normal(100, 15, 1000)
    })
    
    # Test utility functions
    print("Data Info:")
    info = get_data_info(sample_data)
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Test plotting (uncomment to run)
    # plot_data_distribution(sample_data, ['rank', 'score'])
    # create_correlation_matrix(sample_data)
