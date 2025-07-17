"""
Data module for JEE College Prediction project.

Contains web scraping and data preprocessing utilities.
"""

# Import preprocessing (always available)
from .preprocessing import DataPreprocessor

# Import scraper (optional, depends on scrapy)
try:
    from .scraper import JEEDataScraper
except ImportError:
    JEEDataScraper = None
    print("Warning: JEEDataScraper not available. Install scrapy if needed.")

__all__ = ['JEEDataScraper', 'DataPreprocessor']
