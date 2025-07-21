# JEE College Prediction 🎓

A comprehensive machine learning project to predict college admission outcomes based on JEE Main and Advanced results, built with industry-standard practices and comprehensive documentation.

## 🎯 Project Overview

This project helps students predict their chances of getting admission to various engineering colleges based on their JEE (Joint Entrance Examination) rank, gender, and seat type preference. The system uses historical admission data from JOSAA (Joint Seat Allocation Authority) to make accurate predictions using advanced machine learning techniques.

### Key Features 🌟

- **🔍 Data Scraping**: Automated collection of admission data from JOSAA website
- **🧹 Data Preprocessing**: Robust cleaning and preparation of data for machine learning
- **🤖 Machine Learning Model**: Random Forest-based prediction model with multi-output support
- **📊 Comprehensive Analysis**: Detailed exploratory data analysis and visualization
- **📱 Easy to Use**: Simple CLI interface and comprehensive notebooks
- **🚀 Production Ready**: Proper error handling, logging, and deployment considerations

## 🏗️ Project Structure

```
JeeCollegePredictor/
├── src/                       # Source code
│   ├── data/
│   │   ├── __init__.py
│   │   ├── scraper.py         # Web scraping utilities
│   │   ├── preprocessing.py   # Data cleaning and preprocessing
│   │   └── legacy_scraper.py  # Legacy scraping code
│   ├── models/
│   │   ├── __init__.py
│   │   └── predictor.py       # Machine learning model
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py         # Utility functions
│   └── __init__.py
├── data/                      # Data directory (ignored in git)
│   ├── raw/                   # Raw scraped data
│   └── processed/             # Cleaned and processed data
├── models/                    # Saved ML models (ignored in git)
├── notebooks/                 # Jupyter notebooks for analysis
│   ├── 01_data_analysis.ipynb # Comprehensive data analysis
│   ├── comprehensive_analysis.ipynb
│   ├── data_exploration.ipynb
│   ├── feature_engineering.ipynb
│   ├── model_training.ipynb
│   ├── model_validation.ipynb
│   └── test_notebook.ipynb
├── tests/                     # Unit tests
│   └── test_main.py
├── docs/                      # Documentation
│   ├── README.md              # Documentation index
│   ├── api.md                 # API documentation
│   ├── data_schema.md         # Data schema documentation
│   └── model_details.md       # Model details and performance
├── config/
│   └── config.json            # Configuration settings
├── .gitignore                 # Git ignore file (includes data/ and models/)
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── Makefile                   # Development commands
├── main.py                    # CLI interface
├── PROJECT_STATUS.md          # Project status and progress
├── README.md                  # This file
└── LICENSE                    # MIT License
```

## 🚀 Getting Started

### Prerequisites 📋

- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Nikhil-Tummala/Jee-College-Prediction.git
cd JeeCollegePredictor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start

#### 1. Data Scraping

```python
from src.data.scraper import JEEDataScraper
import scrapy.crawler as crawler

# Initialize the scraper
scraper = JEEDataScraper()

# Run the scraper (requires scrapy)
# scrapy crawl jee_data_scraper
```

#### 2. Data Preprocessing

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

#### 3. Model Training

```python
from src.models.predictor import JEECollegePredictor

# Initialize the predictor
predictor = JEECollegePredictor()

# Load processed data
data = predictor.load_data("data/processed/data_v2.pkl")

# Prepare features and train model
X, y = predictor.prepare_features(data)
results = predictor.train_model(X, y)

# Save the trained model
predictor.save_model("models/jee_model.joblib")
```

#### 4. Making Predictions

```python
# Load trained model
predictor = JEECollegePredictor()
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

## 📈 Model Performance

The model uses a Random Forest classifier with the following features:
- **Input Features**: Opening Rank, Gender, Seat Type
- **Target Variables**: Institute, Round
- **Preprocessing**: Standard scaling for numerical features, One-hot encoding for categorical features

## 🔧 Configuration

Edit `config/config.json` to customize:
- Data paths
- Model hyperparameters
- Scraping settings
- Logging configuration

## 📚 Documentation

- [API Documentation](docs/api.md)
- [Data Schema](docs/data_schema.md)
- [Model Details](docs/model_details.md)

## 🧪 Testing

Run tests using pytest:
```bash
pytest tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Tummala Nikhil Phaneendra**
- GitHub: [@Nikhil-Tummala](https://github.com/Nikhil-Tummala)
- LinkedIn: [Nikhil Phaneendra Tummala](https://linkedin.com/in/nikhil-tummala)

**Gunnam Pavan Sri Ram Manikanta**
- GitHub: [@GunnamPavan](https://github.com/gunnampavan19)
- LinkedIn: [Gunnam Pavan Sri Ram Manikanta](https://in.linkedin.com/in/pavan-gunnam-5a790a235)

## 🙏 Acknowledgments

- JOSAA for providing admission data
- scikit-learn community for machine learning tools
- Open source contributors

## 📊 Data Sources

- [JOSAA Official Website](https://josaa.admissions.nic.in/)
- Historical admission data from 2016-2023

## 🔮 Future Enhancements

- [ ] Web interface for predictions
- [ ] Real-time data updates
- [ ] Advanced feature engineering
- [ ] Deep learning models
- [ ] Mobile app development
- [ ] API deployment

## 📞 Support

If you have any questions or need help, please:
1. Check the [Issues](https://github.com/Nikhil-Tummala/Jee-College-Prediction/issues) page
2. Create a new issue if your problem isn't already listed
3. Contact the maintainer directly

---

**Note**: This project is for educational purposes. Always verify predictions with official sources before making important decisions.
