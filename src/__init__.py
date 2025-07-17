"""
JEE College Prediction Project

A machine learning project to predict college admission outcomes
based on JEE Main and Advanced results.
"""

__version__ = "1.0.0"
__author__ = "Tummala Nikhil Phaneendra"
__email__ = "tummala1911@gmail.com"

"""
JEE College Prediction Project

A machine learning project to predict college admission outcomes
based on JEE Main and Advanced results.
"""

__version__ = "1.0.0"
__author__ = "Tummala Nikhil Phaneendra"
__email__ = "tummala1911@gmail.com"

# Import modules with error handling
try:
    from .data import preprocessing
    from .data.preprocessing import DataPreprocessor
except ImportError as e:
    print(f"Warning: Could not import data preprocessing: {e}")
    DataPreprocessor = None

try:
    from .data import scraper
except ImportError as e:
    print(f"Warning: Could not import scraper (install scrapy if needed): {e}")
    scraper = None

try:
    from .models import predictor
    from .models.predictor import JEEPredictor
except ImportError as e:
    print(f"Warning: Could not import predictor: {e}")
    JEEPredictor = None

try:
    from .utils import helpers
except ImportError as e:
    print(f"Warning: Could not import helpers: {e}")
    helpers = None

__all__ = [
    'scraper',
    'preprocessing', 
    'predictor',
    'helpers',
    'DataPreprocessor',
    'JEEPredictor'
]
