# Installation and Usage Guide

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Nikhil-Tummala/JeeCollegePredictor.git
cd JeeCollegePredictor
```

### 2. Set up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install the Package
```bash
pip install -e .
```

## 📊 Data Setup

### Option 1: Use Sample Data
```bash
# Create sample data for testing
python -c "
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

# Create sample data
np.random.seed(42)
data = {
    'Opening Rank': np.random.randint(1, 100000, 1000),
    'Closing Rank': np.random.randint(1, 100000, 1000),
    'Gender': np.random.choice(['Male', 'Female'], 1000),
    'Seat Type': np.random.choice(['General', 'OBC', 'SC', 'ST'], 1000),
    'Institute': np.random.choice(['IIT Delhi', 'IIT Bombay', 'IIT Madras', 'IIT Kanpur'], 1000),
    'round': np.random.randint(1, 7, 1000)
}

df = pd.DataFrame(data)
Path('data/raw').mkdir(parents=True, exist_ok=True)
with open('data/raw/data_v1.pkl', 'wb') as f:
    pickle.dump(df, f)
print('Sample data created!')
"
```

### Option 2: Run Data Scraper
```bash
# Run the web scraper (requires internet connection)
python src/data/scraper.py
```

## 🔧 Usage

### Command Line Interface
```bash
# Show help
python main.py --help

# Train model with sample data
python main.py train --data data/raw/data_v1.pkl

# Make predictions
python main.py predict --rank 5000 --gender Male --seat-type General
```

### Using Jupyter Notebooks
```bash
# Start Jupyter
jupyter notebook

# Open the main analysis notebook
# Navigate to notebooks/01_data_analysis.ipynb
```

### Python API
```python
from src.models.predictor import JEEPredictor

# Initialize predictor
predictor = JEEPredictor()

# Load data and train model
predictor.load_data('data/raw/data_v1.pkl')
predictor.train_model()

# Make predictions
result = predictor.predict(rank=5000, gender='Male', seat_type='General')
print(result)
```

## 🧪 Testing

### Run Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_main.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Using Makefile
```bash
# Install dependencies
make install

# Run tests
make test

# Run linting
make lint

# Clean build files
make clean
```

## 📚 Documentation

### Generate Documentation
```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Generate docs
cd docs
make html
```

### View Documentation
- API Documentation: `docs/api.md`
- Data Schema: `docs/data_schema.md`
- Model Details: `docs/model_details.md`

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure the package is installed
   pip install -e .
   ```

2. **Data File Not Found**
   ```bash
   # Check if data file exists
   ls -la data/raw/
   
   # Create sample data if needed
   python -c "from src.utils.helpers import create_sample_data; create_sample_data()"
   ```

3. **Memory Issues**
   ```bash
   # Reduce data size or use streaming
   python -c "
   import pandas as pd
   df = pd.read_pickle('data/raw/data_v1.pkl')
   df_small = df.sample(n=1000)
   df_small.to_pickle('data/raw/data_small.pkl')
   "
   ```

## 📈 Performance Optimization

### For Large Datasets
```python
# Use chunked processing
import pandas as pd

def process_large_file(filepath, chunksize=10000):
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        # Process chunk
        yield process_chunk(chunk)
```

### Memory Management
```python
# Optimize data types
df['rank'] = df['rank'].astype('int32')
df['category'] = df['category'].astype('category')
```

## 🚀 Deployment

### Local Development Server
```bash
# Start development server
python main.py serve --port 8000
```

### Docker Deployment
```bash
# Build Docker image
docker build -t jee-predictor .

# Run container
docker run -p 8000:8000 jee-predictor
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

See `CONTRIBUTING.md` for detailed guidelines.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
