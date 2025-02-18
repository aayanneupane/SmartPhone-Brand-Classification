from skimage import io, transform, color
from skimage.feature import hog
import numpy as np
import logging
from PIL import Image

class ImageProcessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.logger = logging.getLogger(__name__)

    def validate_image_content(self, image):
        """
        Validate image content for quality control
        """
        if image is None:
            return False, "Image is None"

        if not isinstance(image, np.ndarray):
            return False, "Image is not a numpy array"

        if image.size == 0:
            return False, "Image is empty"

        if len(image.shape) < 2:
            return False, "Invalid image dimensions"

        return True, "Image is valid"

    def load_image(self, image_path):
        """
        Load and preprocess an image with enhanced validation
        """
        try:
            self.logger.debug(f"Loading image from {image_path}")
            try:
                # Try loading with scikit-image first
                image = io.imread(image_path)
                self.logger.debug("Successfully loaded image with scikit-image")
            except Exception as e:
                self.logger.debug(f"scikit-image failed to load {image_path}, trying PIL: {str(e)}")
                # Fallback to PIL
                pil_image = Image.open(image_path)
                image = np.array(pil_image)
                self.logger.debug("Successfully loaded image with PIL")

            # Validate loaded image
            is_valid, message = self.validate_image_content(image)
            if not is_valid:
                raise ValueError(message)

            # Convert to RGB if image is RGBA
            if len(image.shape) == 3 and image.shape[-1] == 4:
                self.logger.debug("Converting RGBA to RGB")
                image = color.rgba2rgb(image)

            # Convert to grayscale
            if len(image.shape) == 3:
                self.logger.debug("Converting to grayscale")
                image = color.rgb2gray(image)

            # Resize image
            self.logger.debug(f"Resizing image to {self.target_size}")
            image = transform.resize(image, self.target_size, mode='constant')

            self.logger.debug(f"Final image shape: {image.shape}")
            return image

        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            raise

    def extract_features(self, image):
        """
        Extract HOG features with validation
        """
        try:
            is_valid, message = self.validate_image_content(image)
            if not is_valid:
                raise ValueError(f"Invalid input image: {message}")

            self.logger.debug("Extracting HOG features")
            features = hog(
                image,
                orientations=9,  # Increased from 8 for better feature detection
                pixels_per_cell=(8, 8),  # Decreased from 16 for more detail
                cells_per_block=(3, 3),  # Increased from 2 for better normalization
                visualize=False
            )

            self.logger.debug(f"Extracted {len(features)} features")
            return features

        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            raise

    def process_image(self, image_path):
        """
        Complete image processing pipeline with enhanced validation
        """
        self.logger.info(f"Processing image: {image_path}")

        image = self.load_image(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        features = self.extract_features(image)
        if features is None or len(features) == 0:
            raise ValueError(f"Failed to extract features from image: {image_path}")

        self.logger.debug(f"Feature vector shape: {features.shape}")
        return features