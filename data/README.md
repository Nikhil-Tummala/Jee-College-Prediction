# Data Directory

This directory contains all data files for the JEE College Predictor project.

## 📁 Directory Structure

```
data/
├── raw/          # Raw, unprocessed data files
├── processed/    # Cleaned and processed data files
├── external/     # External data sources
├── interim/      # Intermediate data files
└── README.md     # This file
```

## 📊 Data Types

### Raw Data (`raw/`)
- **`data_v1.pkl`**: Original scraped data from JOSAA website
- **`*.csv`**: CSV exports of raw data
- **`*.json`**: JSON format data files

### Processed Data (`processed/`)
- **`cleaned_data.pkl`**: Cleaned and preprocessed data
- **`train_data.pkl`**: Training dataset
- **`test_data.pkl`**: Test dataset
- **`validation_data.pkl`**: Validation dataset

### External Data (`external/`)
- **`institute_rankings.csv`**: Institute ranking data
- **`category_mapping.json`**: Category code mappings
- **`state_codes.csv`**: State code mappings

## 🔒 Data Privacy and Security

- All data files are excluded from version control (see `.gitignore`)
- Personal information is anonymized
- Data is stored locally and not shared publicly
- Follow data protection guidelines when handling sensitive information

## 📈 Data Schema

### Main Dataset Schema

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `Opening Rank` | int | Opening rank for admission | 1500 |
| `Closing Rank` | int | Closing rank for admission | 2000 |
| `Gender` | str | Student gender | 'Male', 'Female' |
| `Seat Type` | str | Reservation category | 'General', 'OBC', 'SC', 'ST' |
| `Institute` | str | Institute name | 'IIT Delhi' |
| `Branch` | str | Branch name | 'Computer Science' |
| `Round` | int | Admission round | 1, 2, 3, etc. |
| `Year` | int | Admission year | 2023 |

## 🔄 Data Processing Pipeline

1. **Data Collection**: Web scraping from JOSAA website
2. **Data Cleaning**: Handle missing values, fix data types
3. **Data Validation**: Ensure data quality and consistency
4. **Feature Engineering**: Create new features for modeling
5. **Data Splitting**: Split into train/validation/test sets

## 📋 Data Quality Checks

- **Completeness**: Check for missing values
- **Consistency**: Verify data format consistency
- **Accuracy**: Validate against known sources
- **Uniqueness**: Remove duplicate records
- **Validity**: Ensure values are within expected ranges

## 🚀 Getting Started with Data

### Loading Data
```python
import pandas as pd
import pickle

# Load raw data
with open('data/raw/data_v1.pkl', 'rb') as f:
    raw_data = pickle.load(f)

# Load processed data
processed_data = pd.read_pickle('data/processed/cleaned_data.pkl')
```

### Basic Data Exploration
```python
# Check data shape
print(f"Data shape: {raw_data.shape}")

# Check column types
print(raw_data.dtypes)

# Basic statistics
print(raw_data.describe())

# Check missing values
print(raw_data.isnull().sum())
```

## 📊 Sample Data Generation

If you don't have access to real data, you can generate sample data:

```python
import pandas as pd
import numpy as np
from pathlib import Path

# Create sample data
np.random.seed(42)
sample_data = pd.DataFrame({
    'Opening Rank': np.random.randint(1, 50000, 1000),
    'Closing Rank': np.random.randint(1, 50000, 1000),
    'Gender': np.random.choice(['Male', 'Female'], 1000),
    'Seat Type': np.random.choice(['General', 'OBC', 'SC', 'ST'], 1000),
    'Institute': np.random.choice(['IIT Delhi', 'IIT Bombay', 'IIT Madras'], 1000),
    'round': np.random.randint(1, 7, 1000),
    'year': np.random.choice([2020, 2021, 2022, 2023], 1000)
})

# Save sample data
Path('raw').mkdir(exist_ok=True)
sample_data.to_pickle('raw/sample_data.pkl')
```

## 🔧 Data Utilities

### Data Validation
```python
def validate_data(df):
    """Validate data quality."""
    checks = {
        'non_empty': len(df) > 0,
        'required_columns': all(col in df.columns for col in ['Opening Rank', 'Institute']),
        'valid_ranks': df['Opening Rank'].between(1, 300000).all(),
        'no_nulls_in_key_columns': df[['Opening Rank', 'Institute']].notnull().all().all()
    }
    return checks
```

### Data Cleaning
```python
def clean_data(df):
    """Clean and preprocess data."""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df['Gender'] = df['Gender'].fillna('Unknown')
    
    # Fix data types
    df['Opening Rank'] = df['Opening Rank'].astype(int)
    df['Closing Rank'] = df['Closing Rank'].astype(int)
    
    return df
```

## 📚 Data Sources

- **JOSAA Website**: Primary source for admission data
- **Institute Websites**: Additional institute information
- **Government Portals**: Official category and quota information

## 🔄 Data Updates

- Data is updated annually after each admission cycle
- Incremental updates may be available during admission season
- Version control is maintained for all data updates

## 📞 Support

For data-related issues or questions:
- Check the documentation in `docs/data_schema.md`
- Review the data processing notebooks in `notebooks/`
- Contact the development team for assistance

---

**Note**: This directory is excluded from version control to protect sensitive data and manage large file sizes. Always ensure proper data handling and privacy compliance.
