from skimage import io, transform, color
from skimage.feature import hog
import numpy as np
import logging

class ImageProcessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.logger = logging.getLogger(__name__)

    def load_image(self, image_path):
        """
        Load and preprocess an image
        """
        try:
            # Load image
            image = io.imread(image_path)
            
            # Convert to RGB if image is RGBA
            if image.shape[-1] == 4:
                image = color.rgba2rgb(image)
            
            # Convert to grayscale
            if len(image.shape) == 3:
                image = color.rgb2gray(image)
                
            # Resize image
            image = transform.resize(image, self.target_size)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            raise

    def extract_features(self, image):
        """
        Extract HOG features from the image
        """
        try:
            # Extract HOG features
            features = hog(
                image,
                orientations=8,
                pixels_per_cell=(16, 16),
                cells_per_block=(2, 2),
                visualize=False
            )
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            raise

    def process_image(self, image_path):
        """
        Complete image processing pipeline
        """
        image = self.load_image(image_path)
        features = self.extract_features(image)
        return features
