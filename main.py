import argparse
import logging
from ml_model import SmartphoneClassifier
from image_processor import ImageProcessor
from utils import validate_image_path, format_prediction_output, load_training_data
from logger import setup_logger
import pickle
import os

def main():
    # Set up logging
    logger = setup_logger()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Smartphone Brand Classification'
    )
    parser.add_argument(
        '--train',
        type=str,
        help='Path to training data directory containing brand subdirectories'
    )
    parser.add_argument(
        '--predict',
        type=str,
        help='Path to the smartphone image for classification'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='smartphone_classifier.pkl',
        help='Path to save/load the trained model'
    )

    args = parser.parse_args()

    if not args.train and not args.predict:
        parser.error("Either --train or --predict must be specified")

    try:
        classifier = SmartphoneClassifier()

        # Training mode
        if args.train:
            logger.info("Loading training data...")
            features, labels = load_training_data(args.train)

            logger.info("Training model...")
            train_score, test_score = classifier.train(features, labels)

            logger.info(f"Saving model to {args.model}")
            with open(args.model, 'wb') as f:
                pickle.dump(classifier, f)

            logger.info("Training completed successfully!")

        # Prediction mode
        if args.predict:
            validate_image_path(args.predict)

            # Load trained model if exists
            if os.path.exists(args.model):
                logger.info(f"Loading trained model from {args.model}")
                with open(args.model, 'rb') as f:
                    classifier = pickle.load(f)
            else:
                raise ValueError(f"Trained model not found at {args.model}")

            # Process image and extract features
            processor = ImageProcessor()
            logger.info("Processing image...")
            features = processor.process_image(args.predict)

            # Make prediction
            logger.info("Making prediction...")
            prediction = classifier.predict(features)

            # Display results
            print(format_prediction_output(prediction))

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())