import os
import logging
from pathlib import Path
import numpy as np
from collections import defaultdict

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

    # Additional validation for image integrity
    try:
        from PIL import Image
        img = Image.open(image_path)
        img.verify()  # Verify image integrity
        return True
    except Exception as e:
        raise ValueError(f"Invalid or corrupted image file: {str(e)}")

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
    Load training images from data directory with batch processing
    """
    from image_processor import ImageProcessor

    processor = ImageProcessor()
    features = []
    labels = []
    logger = logging.getLogger(__name__)

    brands = ['iphone', 'samsung', 'pixel']
    processed_counts = defaultdict(lambda: {'success': 0, 'failed': 0})
    batch_size = 8  # Process images in batches for better performance
    max_images_per_brand = 50  # Limit number of images per brand for balanced training

    for brand in brands:
        brand_dir = Path(data_dir) / brand
        if not brand_dir.exists():
            raise ValueError(f"Training directory for {brand} not found: {brand_dir}")

        logger.info(f"Loading {brand} images...")

        # Pre-validate images before processing
        valid_images = []
        for img_path in brand_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    if validate_image_path(str(img_path)):
                        valid_images.append(str(img_path))
                        if len(valid_images) >= max_images_per_brand:
                            break
                except Exception as e:
                    logger.warning(f"Skipping invalid image {img_path}: {str(e)}")
                    processed_counts[brand]['failed'] += 1
                    continue

        # Process valid images in batches
        for i in range(0, len(valid_images), batch_size):
            batch_paths = valid_images[i:i + batch_size]
            try:
                # Process batch of images in parallel
                batch_features = processor.process_images_batch(batch_paths)
                features.extend(batch_features)
                labels.extend([brand] * len(batch_features))
                processed_counts[brand]['success'] += len(batch_features)
            except Exception as e:
                processed_counts[brand]['failed'] += len(batch_paths)
                logger.warning(f"Error processing batch for {brand}: {str(e)}")

    # Log processing summary
    logger.info("\nImage Processing Summary:")
    total_success = 0
    total_failed = 0
    for brand, counts in processed_counts.items():
        success = counts['success']
        failed = counts['failed']
        total_success += success
        total_failed += failed
        logger.info(f"{brand}: Successfully processed {success} images, Failed {failed} images")

    logger.info(f"\nTotal: Successfully processed {total_success} images, Failed {total_failed} images")

    if not features:
        raise ValueError("No valid training images found")

    return np.array(features), np.array(labels)