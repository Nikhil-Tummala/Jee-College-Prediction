# Development Guide

## 🛠️ Development Setup

### Environment Setup
```bash
# Clone repository
git clone https://github.com/Nikhil-Tummala/JeeCollegePredictor.git
cd JeeCollegePredictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Development Tools
```bash
# Code formatting
black src/ tests/
isort src/ tests/

# Linting
flake8 src/ tests/
pylint src/

# Type checking
mypy src/

# Testing
pytest tests/ --cov=src
```

## 📋 Code Style Guidelines

### Python Style
- Follow PEP 8 guidelines
- Use type hints for function signatures
- Document all public functions and classes
- Maximum line length: 88 characters (Black default)

### Example Code Structure
```python
"""Module docstring."""

import logging
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ExampleClass:
    """Example class with proper documentation."""
    
    def __init__(self, param: str) -> None:
        """Initialize the class.
        
        Args:
            param: Description of parameter
        """
        self.param = param
        self._logger = logging.getLogger(__name__)
    
    def public_method(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Public method with type hints and documentation.
        
        Args:
            data: Input dataframe
            
        Returns:
            Dictionary containing results
            
        Raises:
            ValueError: If data is invalid
        """
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        return {"result": "success"}
    
    def _private_method(self) -> None:
        """Private method (starts with underscore)."""
        pass
```

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── __init__.py
├── conftest.py          # Pytest configuration and fixtures
├── test_data/           # Test data files
│   ├── sample_data.pkl
│   └── expected_output.json
├── unit/                # Unit tests
│   ├── test_preprocessing.py
│   ├── test_predictor.py
│   └── test_helpers.py
├── integration/         # Integration tests
│   ├── test_pipeline.py
│   └── test_scraper.py
└── test_main.py        # Main CLI tests
```

### Writing Tests
```python
import pytest
import pandas as pd
from unittest.mock import Mock, patch

from src.data.preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Test suite for DataPreprocessor."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'rank': [1000, 2000, 3000],
            'gender': ['Male', 'Female', 'Male'],
            'institute': ['IIT Delhi', 'IIT Bombay', 'IIT Madras']
        })
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return DataPreprocessor()
    
    def test_clean_data_success(self, preprocessor, sample_data):
        """Test successful data cleaning."""
        result = preprocessor.clean_data(sample_data)
        
        assert not result.empty
        assert result.shape[0] == 3
        assert 'rank' in result.columns
    
    def test_clean_data_empty_input(self, preprocessor):
        """Test handling of empty input."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Data cannot be empty"):
            preprocessor.clean_data(empty_df)
    
    @patch('src.data.preprocessing.logger')
    def test_logging(self, mock_logger, preprocessor, sample_data):
        """Test that logging works correctly."""
        preprocessor.clean_data(sample_data)
        
        mock_logger.info.assert_called()
```

### Test Commands
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_main.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run with verbose output
pytest -v

# Run only failed tests
pytest --lf

# Run tests matching pattern
pytest -k "test_clean_data"
```

## 📦 Package Structure

### Source Code Organization
```
src/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── scraper.py          # Web scraping functionality
│   ├── preprocessing.py    # Data cleaning and preprocessing
│   └── validators.py       # Data validation utilities
├── models/
│   ├── __init__.py
│   ├── predictor.py        # Main prediction model
│   ├── training.py         # Model training utilities
│   └── evaluation.py       # Model evaluation metrics
├── utils/
│   ├── __init__.py
│   ├── helpers.py          # General utility functions
│   ├── logging_config.py   # Logging configuration
│   └── config.py           # Configuration management
└── api/
    ├── __init__.py
    ├── routes.py           # API routes
    └── models.py           # API data models
```

### Adding New Modules
1. Create module file with proper docstring
2. Add to appropriate `__init__.py`
3. Write comprehensive tests
4. Update documentation
5. Add to CI/CD pipeline

## 🔧 Configuration Management

### Configuration Files
```yaml
# config/config.yaml
database:
  host: localhost
  port: 5432
  name: jee_predictor

model:
  type: random_forest
  n_estimators: 100
  random_state: 42

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Environment Variables
```bash
# .env file
DATABASE_URL=postgresql://user:pass@localhost:5432/jee_predictor
SECRET_KEY=your-secret-key
DEBUG=true
```

## 📊 Data Management

### Data Versioning
```bash
# Use DVC for data versioning
dvc init
dvc add data/raw/data_v1.pkl
dvc push

# Track data changes
dvc add data/processed/cleaned_data.pkl
git add data/processed/cleaned_data.pkl.dvc
git commit -m "Add processed data v2"
```

### Data Pipeline
```python
# src/data/pipeline.py
from pathlib import Path
import pandas as pd

class DataPipeline:
    """Data processing pipeline."""
    
    def __init__(self, config: dict):
        self.config = config
        self.raw_data_path = Path(config['data']['raw_path'])
        self.processed_data_path = Path(config['data']['processed_path'])
    
    def run(self) -> None:
        """Run the complete data pipeline."""
        # Extract
        raw_data = self.extract_data()
        
        # Transform
        processed_data = self.transform_data(raw_data)
        
        # Load
        self.load_data(processed_data)
    
    def extract_data(self) -> pd.DataFrame:
        """Extract raw data."""
        return pd.read_pickle(self.raw_data_path)
    
    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform and clean data."""
        # Data cleaning logic
        return data.dropna()
    
    def load_data(self, data: pd.DataFrame) -> None:
        """Save processed data."""
        data.to_pickle(self.processed_data_path)
```

## 🚀 CI/CD Pipeline

### GitHub Actions
```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
      with:
        file: ./coverage.xml
```

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3.9
  
  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
  
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
```

## 📈 Performance Monitoring

### Logging
```python
import logging
from src.utils.logging_config import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Use in code
logger.info("Processing data with shape: %s", data.shape)
logger.warning("Low confidence prediction: %s", confidence)
logger.error("Failed to process data: %s", str(e))
```

### Profiling
```python
import cProfile
import pstats

def profile_function(func):
    """Decorator to profile function performance."""
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        stats = pstats.Stats(pr)
        stats.sort_stats('cumulative')
        stats.print_stats(10)
        return result
    return wrapper
```

## 🔍 Debugging

### Debug Configuration
```python
# debug_config.py
import logging
import pandas as pd

# Set debug mode
DEBUG = True

if DEBUG:
    logging.basicConfig(level=logging.DEBUG)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
```

### Common Debugging Techniques
```python
# Add debug prints
print(f"DEBUG: Data shape: {data.shape}")
print(f"DEBUG: Column types: {data.dtypes}")

# Use pdb for interactive debugging
import pdb; pdb.set_trace()

# Use logging for better debugging
logger.debug("Processing batch %d of %d", batch_idx, total_batches)
```

## 🤝 Contributing

### Git Workflow
```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes and commit
git add .
git commit -m "Add new feature: description"

# Push to remote
git push origin feature/new-feature

# Create pull request
# Review and merge
```

### Code Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests are added and passing
- [ ] Documentation is updated
- [ ] No sensitive data is committed
- [ ] Performance impact is considered
- [ ] Error handling is appropriate

## 📚 Documentation

### API Documentation
```python
def predict_admission(rank: int, gender: str, seat_type: str) -> Dict[str, Any]:
    """Predict admission probability for given parameters.
    
    Args:
        rank: JEE rank (1-300000)
        gender: Student gender ('Male', 'Female', 'Other')
        seat_type: Seat category ('General', 'OBC', 'SC', 'ST')
    
    Returns:
        Dictionary containing:
            - probability: Admission probability (0-1)
            - institutes: List of recommended institutes
            - confidence: Prediction confidence (0-1)
    
    Raises:
        ValueError: If rank is out of valid range
        TypeError: If parameters are not correct types
    
    Example:
        >>> result = predict_admission(5000, 'Male', 'General')
        >>> print(result['probability'])
        0.85
    """
```

### README Templates
- Clear project description
- Installation instructions
- Usage examples
- Contributing guidelines
- License information

This development guide provides comprehensive information for developers working on the JEE College Predictor project. Follow these guidelines to maintain code quality and consistency.
