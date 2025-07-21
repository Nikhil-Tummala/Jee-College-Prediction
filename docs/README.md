# JEE College Prediction - Documentation Index

Welcome to the JEE College Prediction project documentation. This comprehensive guide covers all aspects of the project, from installation to advanced usage.

## 📚 Table of Contents

### Getting Started
- [Quick Start Guide](#quick-start-guide)
- [Installation](#installation)
- [Project Overview](#project-overview)
- [System Requirements](#system-requirements)

### Core Documentation
- [API Reference](api.md) - Complete API documentation
- [Data Schema](data_schema.md) - Data structure and validation
- [Model Details](model_details.md) - Machine learning model specifications

### User Guides
- [User Guide](#user-guide)
- [CLI Usage](#cli-usage)
- [Configuration](#configuration)
- [Examples](#examples)

### Development
- [Development Guide](#development-guide)
- [Testing](#testing)
- [Contributing](#contributing)
- [Architecture](#architecture)

### Advanced Topics
- [Performance Optimization](#performance-optimization)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

---

## Quick Start Guide

### Installation

```bash
# Clone the repository
git clone https://github.com/Nikhil-Tummala/Jee-College-Prediction.git
cd JeeCollegePredictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up the project
make setup-dev
```

### Basic Usage

```bash
# Train the model
python main.py train

# Make a prediction
python main.py predict 1000 Male Open
```

### Python API

```python
from src.models.predictor import JEECollegePredictor

# Load model
predictor = JEECollegePredictor()
predictor.load_model("models/jee_model.joblib")

# Make prediction
result = predictor.predict_single(1000, "Male", "Open")
print(f"Institute: {result['Institute']}, Round: {result['Round']}")
```

## Project Overview

### What is JEE College Prediction?

JEE College Prediction is a machine learning project that predicts college admission outcomes for students based on their JEE (Joint Entrance Examination) rank and other factors. The system uses historical admission data to provide accurate predictions about:

- Which college/institute a student might get admission to
- In which round the admission will be confirmed

### Key Features

- **Accurate Predictions**: Uses Random Forest algorithm for reliable predictions
- **Multi-Output Model**: Predicts both institute and round simultaneously
- **Easy-to-Use API**: Simple Python API and command-line interface
- **Comprehensive Data**: Based on historical JOSAA admission data

### Project Structure

```
JeeCollegePredictor/
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── models/            # ML model implementations
│   └── utils/             # Utility functions
├── data/                  # Data storage
│   ├── raw/              # Raw scraped data
│   └── processed/        # Cleaned data
├── models/               # Trained models
├── notebooks/            # Jupyter notebooks
├── tests/                # Unit tests
├── docs/                 # Documentation
├── config/               # Configuration files
└── main.py              # CLI interface
```

## System Requirements

### Software Requirements

- **Python**: 3.7 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 1GB free space

### Python Dependencies

- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scrapy >= 2.5.0 (for data collection)
- jupyter >= 1.0.0 (for notebooks)

## User Guide

### Data Collection

The project includes a web scraper to collect admission data from JOSAA:

```python
from src.data.scraper import JEEDataScraper
import scrapy

# Configure scraper
scraper = JEEDataScraper(
    start_year=2023,
    end_year=2016,
    max_rounds=6
)

# Run scraper (requires scrapy)
# scrapy crawl jee_data_scraper
```

### Data Processing

Process raw data for model training:

```python
from src.data.preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Load and clean data
raw_data = preprocessor.load_data("data/raw/data_v1.pkl")
cleaned_data = preprocessor.clean_data(raw_data)

# Save processed data
preprocessor.save_processed_data(cleaned_data, "data/processed/data_v2.pkl")
```

### Model Training

Train the machine learning model:

```python
from src.models.predictor import JEECollegePredictor

# Initialize predictor
predictor = JEECollegePredictor()

# Load processed data
data = predictor.load_data("data/processed/data_v2.pkl")

# Prepare features and train
X, y = predictor.prepare_features(data)
results = predictor.train_model(X, y)

# Save trained model
predictor.save_model("models/jee_model.joblib")
```

### Making Predictions

Use the trained model for predictions:

```python
# Load trained model
predictor = JEECollegePredictor()
predictor.load_model("models/jee_model.joblib")

# Single prediction
prediction = predictor.predict_single(
    opening_rank=1000,
    gender="Male",
    seat_type="Open"
)

print(f"Predicted Institute: {prediction['Institute']}")
print(f"Predicted Round: {prediction['Round']}")
```

## CLI Usage

### Training

```bash
# Train with default configuration
python main.py train

# Train with custom configuration
python main.py train --config custom_config.json
```

### Prediction

```bash
# Basic prediction
python main.py predict 1000 Male Open

# Prediction with custom model
python main.py predict 5000 Female SC --model models/custom_model.joblib

# Multiple predictions
python main.py predict 1000 Male Open
python main.py predict 2000 Female SC
python main.py predict 3000 Male OBC
```

### Using Makefile

```bash
# Set up development environment
make setup-dev

# Train model
make train

# Make prediction
make predict RANK=1000 GENDER=Male SEAT_TYPE=Open

# Run tests
make test

# Clean up
make clean
```

## Configuration

### Configuration File

The project uses `config/config.json` for configuration:

```json
{
  "data": {
    "raw_data_path": "data/raw/",
    "processed_data_path": "data/processed/"
  },
  "model": {
    "model_path": "models/",
    "model_name": "jee_model.joblib",
    "random_state": 42,
    "test_size": 0.2,
    "hyperparameters": {
      "n_estimators": 100,
      "max_depth": null
    }
  },
  "features": {
    "input_features": ["Opening Rank", "Gender", "Seat Type"],
    "target_features": ["Institute", "round"]
  }
}
```

### Environment Variables

```bash
# Set data paths
export JEE_DATA_PATH="./data"
export JEE_MODEL_PATH="./models"

# Set logging level
export LOG_LEVEL="INFO"

# Set number of CPU cores for training
export N_JOBS="-1"
```

## Examples

### Complete Workflow Example

```python
import pandas as pd
from src.data.preprocessing import DataPreprocessor
from src.models.predictor import JEECollegePredictor

# Step 1: Load and preprocess data
print("Loading and preprocessing data...")
preprocessor = DataPreprocessor()
raw_data = preprocessor.load_data("data/raw/data_v1.pkl")
cleaned_data = preprocessor.clean_data(raw_data)
preprocessor.save_processed_data(cleaned_data, "data/processed/data_v2.pkl")

# Step 2: Train model
print("Training model...")
predictor = JEECollegePredictor()
X, y = predictor.prepare_features(cleaned_data)
results = predictor.train_model(X, y)
predictor.save_model("models/jee_model.joblib")

# Step 3: Make predictions
print("Making predictions...")
predictions = [
    predictor.predict_single(1000, "Male", "Open"),
    predictor.predict_single(5000, "Female", "SC"),
    predictor.predict_single(10000, "Male", "OBC")
]

for i, pred in enumerate(predictions):
    print(f"Prediction {i+1}: {pred['Institute']}, Round {pred['Round']}")
```

### Batch Prediction Example

```python
import pandas as pd

# Prepare batch data
batch_data = pd.DataFrame({
    'Opening Rank': [1000, 2000, 3000, 4000, 5000],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'Seat Type': ['Open', 'SC', 'OBC', 'ST', 'Open']
})

# Make batch predictions
predictions = predictor.predict(batch_data)

# Create results DataFrame
results = pd.DataFrame({
    'Rank': batch_data['Opening Rank'],
    'Gender': batch_data['Gender'],
    'Seat_Type': batch_data['Seat Type'],
    'Predicted_Institute': predictions['Institute'],
    'Predicted_Round': predictions['round']
})

print(results)
```

## Development Guide

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/Nikhil-Tummala/Jee-College-Prediction.git
cd JeeCollegePredictor

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Set up pre-commit hooks
pre-commit install
```

### Code Style

The project follows PEP 8 style guidelines:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_main.py

# Run specific test
pytest tests/test_main.py::TestJEECollegePredictor::test_predict_single
```

## Testing

### Unit Tests

The project includes comprehensive unit tests:

```python
# Example test
import pytest
from src.models.predictor import JEECollegePredictor

def test_predictor_initialization():
    predictor = JEECollegePredictor()
    assert predictor.random_state == 42
    assert predictor.is_fitted == False

def test_predict_single():
    predictor = JEECollegePredictor()
    # ... setup and train model ...
    
    prediction = predictor.predict_single(1000, "Male", "Open")
    assert 'Institute' in prediction
    assert 'Round' in prediction
```

### Integration Tests

```python
def test_complete_pipeline():
    # Test entire workflow
    preprocessor = DataPreprocessor()
    predictor = JEECollegePredictor()
    
    # Load and process data
    raw_data = load_sample_data()
    cleaned_data = preprocessor.clean_data(raw_data)
    
    # Train model
    X, y = predictor.prepare_features(cleaned_data)
    results = predictor.train_model(X, y, validate=False)
    
    # Make prediction
    prediction = predictor.predict_single(1000, "Male", "Open")
    
    assert results['training_completed'] == True
    assert 'Institute' in prediction
```

## Architecture

### System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │    │   Data Layer    │    │  Model Layer    │
│   (JOSAA Web)   │───▶│  (Scraper &     │───▶│  (ML Pipeline)  │
│                 │    │   Preprocessor) │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐           │
│   User Layer    │    │  Service Layer  │           │
│ (CLI & Python   │◀───│  (Predictor &   │◀──────────┘
│      API)       │    │   Utilities)    │
└─────────────────┘    └─────────────────┘
```

### Module Dependencies

```python
# Core dependencies
src.data.scraper          # Data collection
src.data.preprocessing    # Data cleaning
src.models.predictor     # ML model
src.utils.helpers        # Utilities

# External dependencies
pandas                   # Data manipulation
scikit-learn            # ML algorithms
scrapy                  # Web scraping
numpy                   # Numerical computing
```

## Performance Optimization

### Model Performance

```python
# Optimize Random Forest
model = RandomForestClassifier(
    n_estimators=100,      # Balance accuracy vs speed
    max_depth=None,        # Prevent overfitting
    n_jobs=-1,            # Use all CPU cores
    random_state=42       # Reproducibility
)

# Optimize preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(sparse_output=False), categorical_features)
    ],
    n_jobs=-1  # Parallel preprocessing
)
```

### Memory Optimization

```python
# Optimize data types
df['Opening Rank'] = df['Opening Rank'].astype('int32')
df['round'] = df['round'].astype('int8')

# Use categorical data types
df['Gender'] = df['Gender'].astype('category')
df['Seat Type'] = df['Seat Type'].astype('category')
```

## Deployment

### Local Deployment

```bash
# Install package
pip install -e .

# Run as module
python -m src.main predict 1000 Male Open

# Run as script
python main.py predict 1000 Male Open
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "main.py", "train"]
```

### Cloud Deployment

```python
# Example AWS Lambda function
import json
import joblib
from src.models.predictor import JEECollegePredictor

def lambda_handler(event, context):
    # Load model
    predictor = JEECollegePredictor()
    predictor.load_model("models/jee_model.joblib")
    
    # Extract parameters
    rank = event['rank']
    gender = event['gender']
    seat_type = event['seat_type']
    
    # Make prediction
    prediction = predictor.predict_single(rank, gender, seat_type)
    
    return {
        'statusCode': 200,
        'body': json.dumps(prediction)
    }
```

## Monitoring

### Model Performance Monitoring

```python
import time
import logging

def monitor_prediction(predictor, input_data):
    start_time = time.time()
    
    try:
        prediction = predictor.predict_single(**input_data)
        success = True
        error = None
    except Exception as e:
        prediction = None
        success = False
        error = str(e)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Log metrics
    logging.info(f"Prediction: {success}, Duration: {duration:.3f}s")
    
    return {
        'prediction': prediction,
        'success': success,
        'duration': duration,
        'error': error
    }
```

### System Monitoring

```python
import psutil
import gc

def get_system_metrics():
    return {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
        'python_memory': gc.get_stats()
    }
```

## Troubleshooting

### Common Issues

#### 1. Model Training Fails

**Problem**: Training fails with memory error
**Solution**: 
```python
# Reduce dataset size
data_sample = data.sample(frac=0.8, random_state=42)

# Use smaller model
model = RandomForestClassifier(n_estimators=50, max_depth=10)
```

#### 2. Prediction Errors

**Problem**: Prediction fails with unknown category
**Solution**:
```python
# Use handle_unknown='ignore' in OneHotEncoder
OneHotEncoder(handle_unknown='ignore', sparse_output=False)
```

#### 3. Data Loading Issues

**Problem**: Cannot load pickle file
**Solution**:
```python
# Check file existence
import os
if not os.path.exists("data/processed/data_v2.pkl"):
    print("Data file not found. Please run data preprocessing first.")
    
# Handle version compatibility
import pickle
try:
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)
except:
    # Try with different protocol
    with open("data.pkl", "rb") as f:
        data = pickle.load(f, encoding='latin1')
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Add debug prints
def debug_prediction(predictor, input_data):
    print(f"Input: {input_data}")
    print(f"Model loaded: {predictor.is_fitted}")
    
    # Check preprocessing
    X = pd.DataFrame([input_data])
    X_processed = predictor.model.named_steps['preprocessor'].transform(X)
    print(f"Processed shape: {X_processed.shape}")
    
    # Make prediction
    prediction = predictor.predict_single(**input_data)
    print(f"Prediction: {prediction}")
    
    return prediction
```

## FAQ

### General Questions

**Q: What is the accuracy of the model?**
A: The model typically achieves 75-85% accuracy for institute prediction and 80-90% for round prediction.

**Q: How often should the model be retrained?**
A: Recommend retraining annually with new admission data.

**Q: Can I use this for other entrance exams?**
A: The model is specific to JEE, but the framework can be adapted for other exams.

### Technical Questions

**Q: Why Random Forest over other algorithms?**
A: Random Forest provides good accuracy, interpretability, and robustness for this type of tabular data.

**Q: How to handle missing data?**
A: The preprocessor fills missing Gender values with "Neutral" and removes rows with missing critical data.

**Q: Can I add more features?**
A: Yes, modify the feature configuration in config.json and update the preprocessing pipeline.

## Support

### Getting Help

1. **Documentation**: Check this documentation first
2. **Issues**: Search existing GitHub issues
3. **New Issue**: Create a new issue with detailed description
4. **Discussions**: Use GitHub discussions for questions
5. **Email**: Contact maintainers directly

### Contributing

1. **Fork**: Fork the repository
2. **Branch**: Create a feature branch
3. **Code**: Make your changes
4. **Test**: Run tests and ensure they pass
5. **PR**: Submit a pull request

### License

This project is licensed under the MIT License. See [LICENSE](../LICENSE) for details.

---

## Appendix

### Useful Commands

```bash
# Development commands
make setup-dev          # Set up development environment
make train              # Train model
make test               # Run tests
make clean              # Clean up files
make docs               # Generate documentation

# Git commands
git status              # Check status
git add .               # Stage changes
git commit -m "message" # Commit changes
git push origin main    # Push to remote

# Python commands
python -m venv venv     # Create virtual environment
pip install -e .       # Install in development mode
python -m pytest       # Run tests
python -m black .      # Format code
```

### External Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [Machine Learning Mastery](https://machinelearningmastery.com/)

---

*This documentation is continuously updated. For the latest version, please check the GitHub repository.*
