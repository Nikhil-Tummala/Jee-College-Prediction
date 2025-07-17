# Data Schema Documentation

## Overview

This document describes the data schema for the JEE College Prediction project, including input data formats, processed data structures, and model outputs.

## Raw Data Schema

### Input Data Sources

The project uses data scraped from the JOSAA (Joint Seat Allocation Authority) website. The raw data is stored in pickle format.

### Raw Data Structure

```python
{
    'Institute': str,           # Name of the educational institute
    'Opening Rank': int/str,    # Opening rank for admission
    'Closing Rank': int/str,    # Closing rank for admission
    'Gender': str,              # Gender category (Male/Female/Neutral)
    'Seat Type': str,           # Seat type (Open/SC/ST/OBC/EWS)
    'round': int,               # Admission round number (1-6)
    'year': int,                # Year of admission
    'Branch': str,              # Academic branch/program
    'Category': str,            # Additional category information
    'Quota': str,               # Quota type (if applicable)
    'Pool': str                 # Pool information (if applicable)
}
```

### Field Descriptions

#### Institute
- **Type**: String
- **Description**: Name of the educational institute
- **Examples**: "IIT Delhi", "IIT Bombay", "IIT Madras"
- **Constraints**: Cannot be null/empty
- **Processing**: Used as target variable after label encoding

#### Opening Rank
- **Type**: Integer/String
- **Description**: Minimum rank required for admission
- **Range**: 1 to 500,000+
- **Examples**: 1, 100, "1000", "5000K"
- **Constraints**: Must be positive integer
- **Processing**: Cleaned to remove non-numeric characters, converted to int

#### Closing Rank
- **Type**: Integer/String
- **Description**: Maximum rank for admission
- **Range**: 1 to 500,000+
- **Examples**: 10, 500, "2000", "10000K"
- **Constraints**: Must be positive integer, typically >= Opening Rank
- **Processing**: Cleaned to remove non-numeric characters, converted to int

#### Gender
- **Type**: String
- **Description**: Gender category for admission
- **Valid Values**: "Male", "Female", "Neutral"
- **Examples**: "Male", "Female", null
- **Constraints**: Limited to predefined values
- **Processing**: Null values filled with "Neutral"

#### Seat Type
- **Type**: String
- **Description**: Reservation category for seat
- **Valid Values**: "Open", "SC", "ST", "OBC", "EWS"
- **Examples**: "Open", "SC", "ST", "OBC"
- **Constraints**: Limited to predefined values
- **Processing**: Used as categorical feature

#### Round
- **Type**: Integer
- **Description**: Admission round number
- **Range**: 1 to 6
- **Examples**: 1, 2, 3, 4, 5, 6
- **Constraints**: Must be between 1 and 6
- **Processing**: Used as target variable

#### Year
- **Type**: Integer
- **Description**: Year of admission
- **Range**: 2016 to current year
- **Examples**: 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023
- **Constraints**: Must be valid year
- **Processing**: Used for train/test split

## Processed Data Schema

### Cleaned Data Structure

After preprocessing, the data structure is standardized:

```python
{
    'Institute': str,           # Cleaned institute name
    'Opening Rank': int,        # Cleaned opening rank
    'Closing Rank': int,        # Cleaned closing rank
    'Gender': str,              # Standardized gender (no nulls)
    'Seat Type': str,           # Standardized seat type
    'round': int,               # Round number
    'year': int                 # Year (if available)
}
```

### Feature Engineering

#### Input Features (X)
```python
{
    'Opening Rank': int,        # Primary numeric feature
    'Gender': str,              # Categorical feature
    'Seat Type': str            # Categorical feature
}
```

#### Target Variables (y)
```python
{
    'Institute': int,           # Label encoded institute
    'round': int                # Round number
}
```

### Data Preprocessing Steps

1. **Missing Value Handling**
   - Institute: Rows with missing values are dropped
   - Gender: Missing values filled with "Neutral"
   - Ranks: Invalid values converted to NaN and rows dropped

2. **Data Type Conversion**
   - Opening Rank: Converted to integer
   - Closing Rank: Converted to integer
   - Categorical variables: Kept as strings

3. **Data Cleaning**
   - Rank values: Remove trailing characters (K, L, etc.)
   - String normalization: Consistent casing and formatting

4. **Feature Encoding**
   - Institute: Label encoding (0, 1, 2, ...)
   - Gender: One-hot encoding during model training
   - Seat Type: One-hot encoding during model training
   - Opening Rank: Standard scaling during model training

## Model Input/Output Schema

### Model Input Format

The model expects input in the following format:

```python
pd.DataFrame({
    'Opening Rank': [int],      # List of ranks
    'Gender': [str],            # List of gender values
    'Seat Type': [str]          # List of seat types
})
```

**Example:**
```python
input_data = pd.DataFrame({
    'Opening Rank': [1000, 5000, 10000],
    'Gender': ['Male', 'Female', 'Male'],
    'Seat Type': ['Open', 'SC', 'OBC']
})
```

### Model Output Format

The model returns predictions in the following format:

```python
{
    'Institute': [str],         # List of predicted institutes
    'round': [int]              # List of predicted rounds
}
```

**Example:**
```python
predictions = {
    'Institute': ['IIT Delhi', 'IIT Bombay', 'IIT Madras'],
    'round': [1, 2, 3]
}
```

### Single Prediction Format

For single predictions, the input and output are simplified:

**Input:**
```python
{
    'opening_rank': int,
    'gender': str,
    'seat_type': str
}
```

**Output:**
```python
{
    'Institute': str,
    'Round': int
}
```

## Data Validation Rules

### Input Validation

1. **Opening Rank**
   - Must be positive integer
   - Range: 1 to 500,000
   - Cannot be null/empty

2. **Gender**
   - Must be one of: "Male", "Female", "Neutral"
   - Case-sensitive
   - Null values converted to "Neutral"

3. **Seat Type**
   - Must be one of: "Open", "SC", "ST", "OBC", "EWS"
   - Case-sensitive
   - Cannot be null/empty

4. **Institute**
   - Must be non-empty string
   - Valid institute names from training data

5. **Round**
   - Must be integer between 1 and 6
   - Cannot be null/empty

### Data Quality Checks

1. **Consistency Checks**
   - Opening Rank ≤ Closing Rank (when both present)
   - Year must be reasonable (2016-2030)
   - Round must be valid (1-6)

2. **Completeness Checks**
   - Required fields cannot be null
   - Minimum number of records for training

3. **Range Checks**
   - Ranks within expected ranges
   - Categorical values from predefined lists

## Configuration Schema

### config.json Structure

```json
{
  "data": {
    "raw_data_path": "string",
    "processed_data_path": "string",
    "file_formats": {
      "input": "string",
      "output": "string"
    }
  },
  "features": {
    "input_features": ["string"],
    "target_features": ["string"],
    "categorical_features": ["string"],
    "numerical_features": ["string"]
  },
  "model": {
    "model_path": "string",
    "model_name": "string",
    "random_state": "integer",
    "test_size": "float",
    "hyperparameters": {
      "n_estimators": "integer",
      "max_depth": "integer|null",
      "min_samples_split": "integer",
      "min_samples_leaf": "integer"
    }
  },
  "preprocessing": {
    "handle_missing_values": "boolean",
    "missing_value_strategy": {
      "Gender": "string",
      "numeric_columns": "string"
    },
    "outlier_removal": {
      "enabled": "boolean",
      "method": "string",
      "columns": ["string"]
    }
  }
}
```

## Error Handling Schema

### Error Response Format

```python
{
    "error": True,
    "message": "string",
    "type": "string",
    "details": {
        "field": "string",
        "value": "any",
        "expected": "string"
    }
}
```

### Common Error Types

1. **ValidationError**: Invalid input data
2. **DataError**: Data quality issues
3. **ModelError**: Model-related errors
4. **FileError**: File I/O errors

## Version History

### v1.0.0
- Initial data schema definition
- Basic validation rules
- Core features and targets

### v1.1.0
- Added comprehensive validation
- Enhanced error handling
- Improved documentation

### v1.2.0
- Added configuration schema
- Extended field descriptions
- Added data quality checks

## Usage Examples

### Loading and Validating Data

```python
from src.data.preprocessing import DataPreprocessor

# Load data
preprocessor = DataPreprocessor()
raw_data = preprocessor.load_data("data/raw/data_v1.pkl")

# Validate schema
required_columns = ['Institute', 'Opening Rank', 'Gender', 'Seat Type']
if all(col in raw_data.columns for col in required_columns):
    print("Schema validation passed")
else:
    print("Schema validation failed")

# Clean data
cleaned_data = preprocessor.clean_data(raw_data)
```

### Preparing Model Input

```python
# Prepare features
X = cleaned_data[['Opening Rank', 'Gender', 'Seat Type']]
y = cleaned_data[['Institute', 'round']]

# Validate input schema
assert X.shape[1] == 3, "Input must have 3 features"
assert y.shape[1] == 2, "Target must have 2 variables"
```

### Making Predictions

```python
from src.models.predictor import JEECollegePredictor

# Initialize predictor
predictor = JEECollegePredictor()
predictor.load_model("models/jee_model.joblib")

# Prepare input
input_data = {
    'opening_rank': 1000,
    'gender': 'Male',
    'seat_type': 'Open'
}

# Validate input
assert 1 <= input_data['opening_rank'] <= 500000, "Invalid rank"
assert input_data['gender'] in ['Male', 'Female', 'Neutral'], "Invalid gender"
assert input_data['seat_type'] in ['Open', 'SC', 'ST', 'OBC'], "Invalid seat type"

# Make prediction
prediction = predictor.predict_single(**input_data)
print(f"Predicted Institute: {prediction['Institute']}")
print(f"Predicted Round: {prediction['Round']}")
```

## Best Practices

1. **Data Validation**
   - Always validate input data before processing
   - Use type hints for better code documentation
   - Implement comprehensive error handling

2. **Data Storage**
   - Use consistent file formats (pickle for internal use)
   - Maintain data versioning
   - Document data transformations

3. **Schema Evolution**
   - Plan for schema changes
   - Maintain backward compatibility
   - Version control schema definitions

4. **Performance**
   - Use appropriate data types
   - Optimize for memory usage
   - Consider data partitioning for large datasets

## Support

For schema-related questions:
- Check the code documentation
- Review test cases for examples
- Contact the development team
- Submit issues on GitHub
