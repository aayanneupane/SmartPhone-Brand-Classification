import os
import logging
from pathlib import Path
import numpy as np

def validate_image_path(image_path):
    """
    Validate if the image path exists and has valid extension
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    valid_extensions = {'.jpg', '.jpeg', '.png'}
    file_extension = os.path.splitext(image_path)[1].lower()

    if file_extension not in valid_extensions:
        raise ValueError(
            f"Invalid file extension: {file_extension}. "
            f"Supported extensions: {', '.join(valid_extensions)}"
        )

def format_prediction_output(prediction):
    """
    Format the prediction results for console output
    """
    output = [
        "\nPrediction Results:",
        f"Predicted Brand: {prediction['brand'].upper()}",
        f"Confidence: {prediction['confidence']*100:.2f}%",
        "\nProbabilities for each brand:"
    ]

    for brand, prob in prediction['probabilities'].items():
        output.append(f"- {brand.upper()}: {prob*100:.2f}%")

    return "\n".join(output)

def load_training_data(data_dir):
    """
    Load training images from data directory
    Expected structure:
    data_dir/
        iphone/
            image1.jpg
            image2.jpg
        samsung/
            image1.jpg
            image2.jpg
        pixel/
            image1.jpg
            image2.jpg
    """
    from image_processor import ImageProcessor

    processor = ImageProcessor()
    features = []
    labels = []
    logger = logging.getLogger(__name__)

    brands = ['iphone', 'samsung', 'pixel']  # Focus on original three brands

    for brand in brands:
        brand_dir = Path(data_dir) / brand
        if not brand_dir.exists():
            raise ValueError(f"Training directory for {brand} not found: {brand_dir}")

        logger.info(f"Loading {brand} images...")
        for img_path in brand_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    image_features = processor.process_image(str(img_path))
                    features.append(image_features)
                    labels.append(brand)
                except Exception as e:
                    logger.warning(f"Skipping {img_path}: {str(e)}")

    if not features:
        raise ValueError("No valid training images found")

    return np.array(features), np.array(labels)