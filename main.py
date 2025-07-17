"""
Main script for JEE College Prediction project.

This script provides a command-line interface for the JEE College Prediction system.
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.preprocessing import DataPreprocessor
from src.models.predictor import JEECollegePredictor
from src.utils.helpers import setup_logging, load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(config_path: str = "config/config.json"):
    """
    Train the JEE College Prediction model.
    
    Args:
        config_path (str): Path to configuration file
    """
    logger.info("Starting model training...")
    
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Initialize components
        preprocessor = DataPreprocessor()
        predictor = JEECollegePredictor(
            random_state=config['model']['random_state']
        )
        
        # Load and preprocess data
        raw_data_path = config['data']['raw_data_path'] + "data_v1.pkl"
        processed_data_path = config['data']['processed_data_path'] + "data_v2.pkl"
        
        logger.info(f"Loading data from {raw_data_path}")
        raw_data = preprocessor.load_data(raw_data_path)
        
        logger.info("Cleaning and preprocessing data...")
        cleaned_data = preprocessor.clean_data(raw_data)
        
        if preprocessor.validate_data(cleaned_data):
            preprocessor.save_processed_data(cleaned_data, processed_data_path)
            
            # Prepare features and train model
            X, y = predictor.prepare_features(cleaned_data)
            results = predictor.train_model(
                X, y, 
                test_size=config['model']['test_size']
            )
            
            # Save model
            model_path = config['model']['model_path'] + config['model']['model_name']
            predictor.save_model(model_path)
            
            logger.info("Model training completed successfully!")
            logger.info(f"Model saved to {model_path}")
            
            # Print results
            for key, value in results.items():
                if key.endswith('_accuracy'):
                    logger.info(f"{key}: {value:.4f}")
        else:
            logger.error("Data validation failed!")
            
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise


def predict(rank: int, gender: str, seat_type: str, 
           model_path: str = "models/jee_model.joblib"):
    """
    Make a prediction for a single student.
    
    Args:
        rank (int): Student's JEE rank
        gender (str): Student's gender
        seat_type (str): Seat type preference
        model_path (str): Path to trained model
    """
    try:
        # Load model
        predictor = JEECollegePredictor()
        predictor.load_model(model_path)
        
        # Make prediction
        prediction = predictor.predict_single(
            opening_rank=rank,
            gender=gender,
            seat_type=seat_type
        )
        
        print(f"\nPrediction for Rank {rank}, Gender: {gender}, Seat Type: {seat_type}")
        print(f"Predicted Institute: {prediction['Institute']}")
        print(f"Predicted Round: {prediction['Round']}")
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise


def main():
    """
    Main function to handle command-line arguments and execute appropriate actions.
    """
    parser = argparse.ArgumentParser(
        description="JEE College Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s train                              # Train the model
  %(prog)s predict 1000 Male Open            # Make a prediction
  %(prog)s predict 5000 Female SC            # Make a prediction for SC category
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument(
        '--config', 
        default='config/config.json',
        help='Path to configuration file'
    )
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make a prediction')
    predict_parser.add_argument('rank', type=int, help='JEE rank')
    predict_parser.add_argument('gender', help='Gender (Male/Female)')
    predict_parser.add_argument('seat_type', help='Seat type (Open/SC/ST/OBC)')
    predict_parser.add_argument(
        '--model', 
        default='models/jee_model.joblib',
        help='Path to trained model'
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args.config)
    elif args.command == 'predict':
        predict(args.rank, args.gender, args.seat_type, args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
