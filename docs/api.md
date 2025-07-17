# API Documentation

## Overview

The JEE College Prediction API provides programmatic access to the machine learning model for predicting college admission outcomes based on JEE ranks and other factors.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.models.predictor import JEECollegePredictor

# Initialize the predictor
predictor = JEECollegePredictor()

# Load trained model
predictor.load_model("models/jee_model.joblib")

# Make a prediction
prediction = predictor.predict_single(
    opening_rank=1000,
    gender="Male",
    seat_type="Open"
)

print(f"Predicted Institute: {prediction['Institute']}")
print(f"Predicted Round: {prediction['Round']}")
```

## API Reference

### JEECollegePredictor

The main class for making predictions about JEE college admissions.

#### Constructor

```python
JEECollegePredictor(random_state=42)
```

**Parameters:**
- `random_state` (int): Random state for reproducibility. Default: 42

#### Methods

##### `load_data(file_path)`

Load processed data from pickle file.

**Parameters:**
- `file_path` (str): Path to the pickle file

**Returns:**
- `pd.DataFrame`: Loaded dataframe

**Example:**
```python
data = predictor.load_data("data/processed/data_v2.pkl")
```

##### `prepare_features(df)`

Prepare features and target variables from dataframe.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe

**Returns:**
- `Tuple[pd.DataFrame, pd.DataFrame]`: Features and targets

**Example:**
```python
X, y = predictor.prepare_features(data)
```

##### `build_model(n_estimators=100)`

Build the machine learning pipeline.

**Parameters:**
- `n_estimators` (int): Number of trees in the random forest. Default: 100

**Returns:**
- `Pipeline`: Scikit-learn pipeline

**Example:**
```python
model = predictor.build_model(n_estimators=200)
```

##### `train_model(X, y, test_size=0.2, validate=True)`

Train the machine learning model.

**Parameters:**
- `X` (pd.DataFrame): Features
- `y` (pd.DataFrame): Targets
- `test_size` (float): Test set size for validation. Default: 0.2
- `validate` (bool): Whether to perform validation. Default: True

**Returns:**
- `Dict[str, Any]`: Training results

**Example:**
```python
results = predictor.train_model(X, y, test_size=0.3)
```

##### `predict(X)`

Make predictions using the trained model.

**Parameters:**
- `X` (pd.DataFrame): Features for prediction

**Returns:**
- `Dict[str, Any]`: Predictions

**Example:**
```python
predictions = predictor.predict(X_test)
```

##### `predict_single(opening_rank, gender, seat_type)`

Make prediction for a single student.

**Parameters:**
- `opening_rank` (int): Student's rank
- `gender` (str): Student's gender ("Male", "Female", "Neutral")
- `seat_type` (str): Seat type preference ("Open", "SC", "ST", "OBC")

**Returns:**
- `Dict[str, str]`: Predicted institute and round

**Example:**
```python
prediction = predictor.predict_single(
    opening_rank=1000,
    gender="Male",
    seat_type="Open"
)
```

##### `save_model(file_path)`

Save the trained model.

**Parameters:**
- `file_path` (str): Path to save the model

**Example:**
```python
predictor.save_model("models/my_model.joblib")
```

##### `load_model(file_path)`

Load a pre-trained model.

**Parameters:**
- `file_path` (str): Path to the saved model

**Example:**
```python
predictor.load_model("models/jee_model.joblib")
```

##### `get_feature_importance()`

Get feature importance from the trained model.

**Returns:**
- `pd.DataFrame`: Feature importance scores

**Example:**
```python
importance = predictor.get_feature_importance()
```

### DataPreprocessor

Class for preprocessing JEE admission data.

#### Constructor

```python
DataPreprocessor()
```

#### Methods

##### `load_data(file_path)`

Load data from a pickle file.

**Parameters:**
- `file_path` (str): Path to the pickle file

**Returns:**
- `pd.DataFrame`: Loaded dataframe

##### `clean_data(df)`

Clean the raw data.

**Parameters:**
- `df` (pd.DataFrame): Raw dataframe

**Returns:**
- `pd.DataFrame`: Cleaned dataframe

##### `save_processed_data(df, file_path)`

Save processed data to a pickle file.

**Parameters:**
- `df` (pd.DataFrame): Processed dataframe
- `file_path` (str): Output file path

##### `validate_data(df)`

Validate the processed data.

**Parameters:**
- `df` (pd.DataFrame): Dataframe to validate

**Returns:**
- `bool`: True if validation passes, False otherwise

### JEEDataScraper

Scrapy spider for scraping JEE admission data.

#### Constructor

```python
JEEDataScraper(start_year=2023, end_year=2016, max_rounds=6)
```

**Parameters:**
- `start_year` (int): Starting year for data collection. Default: 2023
- `end_year` (int): Ending year for data collection. Default: 2016
- `max_rounds` (int): Maximum number of rounds to scrape per year. Default: 6

## Command Line Interface

### Training the Model

```bash
python main.py train [--config CONFIG_PATH]
```

**Options:**
- `--config`: Path to configuration file (default: config/config.json)

### Making Predictions

```bash
python main.py predict RANK GENDER SEAT_TYPE [--model MODEL_PATH]
```

**Arguments:**
- `RANK`: JEE rank (integer)
- `GENDER`: Gender (Male/Female/Neutral)
- `SEAT_TYPE`: Seat type (Open/SC/ST/OBC)

**Options:**
- `--model`: Path to trained model (default: models/jee_model.joblib)

**Examples:**
```bash
python main.py predict 1000 Male Open
python main.py predict 5000 Female SC
python main.py predict 10000 Male OBC --model models/custom_model.joblib
```

## Configuration

The system uses a JSON configuration file located at `config/config.json`. Key configuration options include:

### Data Configuration

```json
{
  "data": {
    "raw_data_path": "data/raw/",
    "processed_data_path": "data/processed/",
    "file_formats": {
      "input": "pkl",
      "output": "pkl"
    }
  }
}
```

### Model Configuration

```json
{
  "model": {
    "model_path": "models/",
    "model_name": "jee_model.joblib",
    "random_state": 42,
    "test_size": 0.2,
    "hyperparameters": {
      "n_estimators": 100,
      "max_depth": null,
      "min_samples_split": 2,
      "min_samples_leaf": 1
    }
  }
}
```

### Feature Configuration

```json
{
  "features": {
    "input_features": [
      "Opening Rank",
      "Gender", 
      "Seat Type"
    ],
    "target_features": [
      "Institute",
      "round"
    ],
    "categorical_features": [
      "Gender",
      "Seat Type"
    ],
    "numerical_features": [
      "Opening Rank"
    ]
  }
}
```

## Error Handling

The API includes comprehensive error handling:

### Common Errors

1. **FileNotFoundError**: Data or model file not found
2. **ValueError**: Invalid input parameters
3. **KeyError**: Missing required columns in data
4. **ImportError**: Missing dependencies

### Error Response Format

```python
{
    "error": True,
    "message": "Error description",
    "type": "ErrorType"
}
```

## Performance Considerations

- Model training time depends on dataset size and complexity
- Prediction time is typically under 1 second for single predictions
- Memory usage scales with dataset size
- Use `n_jobs=-1` for parallel processing in Random Forest

## Examples

### Complete Workflow Example

```python
import pandas as pd
from src.data.preprocessing import DataPreprocessor
from src.models.predictor import JEECollegePredictor

# Step 1: Load and preprocess data
preprocessor = DataPreprocessor()
raw_data = preprocessor.load_data("data/raw/data_v1.pkl")
cleaned_data = preprocessor.clean_data(raw_data)
preprocessor.save_processed_data(cleaned_data, "data/processed/data_v2.pkl")

# Step 2: Train model
predictor = JEECollegePredictor()
X, y = predictor.prepare_features(cleaned_data)
results = predictor.train_model(X, y)
predictor.save_model("models/jee_model.joblib")

# Step 3: Make predictions
prediction = predictor.predict_single(1000, "Male", "Open")
print(f"Institute: {prediction['Institute']}")
print(f"Round: {prediction['Round']}")
```

### Batch Prediction Example

```python
# Load model
predictor = JEECollegePredictor()
predictor.load_model("models/jee_model.joblib")

# Prepare batch data
batch_data = pd.DataFrame({
    'Opening Rank': [1000, 2000, 3000],
    'Gender': ['Male', 'Female', 'Male'],
    'Seat Type': ['Open', 'SC', 'OBC']
})

# Make batch predictions
predictions = predictor.predict(batch_data)
print(predictions)
```

## Support

For issues and questions:
- Check the [GitHub Issues](https://github.com/Nikhil-Tummala/Jee-College-Prediction/issues)
- Create a new issue if needed
- Contact the maintainer directly

## Version History

- **v1.0.0**: Initial release with basic prediction functionality
- **v1.1.0**: Added batch prediction support
- **v1.2.0**: Improved error handling and documentation
