"""
Test module for JEE College Prediction project.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import pickle

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.preprocessing import DataPreprocessor
from src.models.predictor import JEECollegePredictor
from src.utils.helpers import get_data_info, validate_data_columns


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
        
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'Institute': ['IIT Delhi', 'IIT Bombay', 'IIT Madras', None, 'IIT Kanpur'],
            'Opening Rank': [1, 2, 3, 4, 5],
            'Closing Rank': [10, 20, 30, 40, 50],
            'Gender': ['Male', 'Female', None, 'Male', 'Female'],
            'Seat Type': ['Open', 'Open', 'SC', 'ST', 'OBC'],
            'round': [1, 1, 2, 2, 3]
        })
    
    def test_clean_rank_data(self):
        """Test rank data cleaning function."""
        # Test various input formats
        assert self.preprocessor.clean_rank_data(100) == 100
        assert self.preprocessor.clean_rank_data("100") == 100
        assert self.preprocessor.clean_rank_data("100.0") == 100
        assert self.preprocessor.clean_rank_data("100K") == 100
        assert pd.isna(self.preprocessor.clean_rank_data("invalid"))
    
    def test_clean_data(self):
        """Test data cleaning functionality."""
        cleaned_data = self.preprocessor.clean_data(self.sample_data)
        
        # Check that rows with missing Institute are removed
        assert len(cleaned_data) == 4
        assert cleaned_data['Institute'].isnull().sum() == 0
        
        # Check that missing Gender values are filled with "Neutral"
        assert cleaned_data['Gender'].isnull().sum() == 0
        assert 'Neutral' in cleaned_data['Gender'].values
    
    def test_validate_data(self):
        """Test data validation."""
        # Test with valid data
        valid_data = self.preprocessor.clean_data(self.sample_data)
        assert self.preprocessor.validate_data(valid_data) == True
        
        # Test with invalid data (missing required columns)
        invalid_data = pd.DataFrame({'col1': [1, 2, 3]})
        assert self.preprocessor.validate_data(invalid_data) == False
    
    def test_get_data_summary(self):
        """Test data summary generation."""
        summary = self.preprocessor.get_data_summary(self.sample_data)
        
        assert 'shape' in summary
        assert 'columns' in summary
        assert 'dtypes' in summary
        assert 'null_counts' in summary
        assert summary['shape'] == (5, 6)


class TestJEECollegePredictor:
    """Test cases for JEECollegePredictor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.predictor = JEECollegePredictor()
        
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'Opening Rank': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 
                      'Female', 'Male', 'Female', 'Male', 'Female'],
            'Seat Type': ['Open', 'Open', 'SC', 'ST', 'OBC', 
                         'Open', 'SC', 'ST', 'OBC', 'Open'],
            'Institute': ['IIT Delhi', 'IIT Bombay', 'IIT Madras', 'IIT Kanpur', 'IIT Kharagpur',
                         'IIT Roorkee', 'IIT Guwahati', 'IIT Hyderabad', 'IIT Indore', 'IIT Mandi'],
            'round': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        })
    
    def test_prepare_features(self):
        """Test feature preparation."""
        X, y = self.predictor.prepare_features(self.sample_data)
        
        assert X.shape[1] == 3  # 3 feature columns
        assert y.shape[1] == 2  # 2 target columns
        assert len(X) == len(y)
    
    def test_build_model(self):
        """Test model building."""
        model = self.predictor.build_model()
        
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    def test_train_model(self):
        """Test model training."""
        X, y = self.predictor.prepare_features(self.sample_data)
        results = self.predictor.train_model(X, y, validate=False)
        
        assert results['training_completed'] == True
        assert self.predictor.is_fitted == True
    
    def test_predict_single(self):
        """Test single prediction."""
        # Train model first
        X, y = self.predictor.prepare_features(self.sample_data)
        self.predictor.train_model(X, y, validate=False)
        
        # Make prediction
        prediction = self.predictor.predict_single(
            opening_rank=1000,
            gender="Male",
            seat_type="Open"
        )
        
        assert 'Institute' in prediction
        assert 'Round' in prediction
        assert isinstance(prediction['Institute'], str)


class TestHelpers:
    """Test cases for utility functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e'],
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
    
    def test_get_data_info(self):
        """Test data info generation."""
        info = get_data_info(self.sample_data)
        
        assert 'shape' in info
        assert 'columns' in info
        assert 'dtypes' in info
        assert 'null_counts' in info
        assert info['shape'] == (5, 3)
    
    def test_validate_data_columns(self):
        """Test data column validation."""
        required_columns = ['col1', 'col2']
        assert validate_data_columns(self.sample_data, required_columns) == True
        
        required_columns = ['col1', 'col2', 'missing_col']
        assert validate_data_columns(self.sample_data, required_columns) == False


class TestIntegration:
    """Integration tests for the entire pipeline."""
    
    def test_full_pipeline(self):
        """Test the complete data processing and prediction pipeline."""
        # Create sample data
        sample_data = pd.DataFrame({
            'Institute': ['IIT Delhi', 'IIT Bombay', 'IIT Madras', 'IIT Kanpur', 'IIT Kharagpur'] * 4,
            'Opening Rank': list(range(1, 21)),
            'Closing Rank': list(range(10, 30)),
            'Gender': ['Male', 'Female'] * 10,
            'Seat Type': ['Open', 'SC', 'ST', 'OBC', 'Open'] * 4,
            'round': [1, 2, 3, 4, 5] * 4
        })
        
        # Step 1: Data preprocessing
        preprocessor = DataPreprocessor()
        cleaned_data = preprocessor.clean_data(sample_data)
        
        # Step 2: Model training
        predictor = JEECollegePredictor()
        X, y = predictor.prepare_features(cleaned_data)
        results = predictor.train_model(X, y, validate=False)
        
        # Step 3: Make predictions
        prediction = predictor.predict_single(
            opening_rank=5,
            gender="Male",
            seat_type="Open"
        )
        
        # Verify results
        assert results['training_completed'] == True
        assert 'Institute' in prediction
        assert 'Round' in prediction


if __name__ == "__main__":
    pytest.main([__file__])
