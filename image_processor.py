from skimage import io, transform, color
from skimage.feature import hog
import numpy as np
import logging
from PIL import Image
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import cv2

class ImageProcessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.logger = logging.getLogger(__name__)
        self.thread_pool = ThreadPoolExecutor(max_workers=2)  # Reduced from 4 to 2 workers

    @lru_cache(maxsize=128)
    def load_image(self, image_path):
        """
        Load and preprocess an image with caching and enhanced validation
        """
        try:
            self.logger.debug(f"Loading image from {image_path}")
            try:
                # Try loading with OpenCV first (faster than scikit-image)
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.logger.debug("Successfully loaded image with OpenCV")
                else:
                    # Fallback to PIL
                    pil_image = Image.open(image_path)
                    image = np.array(pil_image)
                    self.logger.debug("Successfully loaded image with PIL")

            except Exception as e:
                self.logger.debug(f"OpenCV failed to load {image_path}, trying PIL: {str(e)}")
                # Final fallback to PIL
                pil_image = Image.open(image_path)
                image = np.array(pil_image)
                self.logger.debug("Successfully loaded image with PIL")

            # Validate loaded image
            if not self.validate_image_content(image)[0]:
                raise ValueError("Invalid image content")

            # Convert to RGB if image is RGBA
            if len(image.shape) == 3 and image.shape[-1] == 4:
                self.logger.debug("Converting RGBA to RGB")
                image = color.rgba2rgb(image)

            # Convert to grayscale
            if len(image.shape) == 3:
                self.logger.debug("Converting to grayscale")
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Resize image using OpenCV (faster than skimage)
            self.logger.debug(f"Resizing image to {self.target_size}")
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

            # Normalize to [0, 1] range
            image = image.astype(np.float32) / 255.0

            self.logger.debug(f"Final image shape: {image.shape}")
            return image

        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            raise

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

    def extract_features(self, image):
        """
        Extract optimized HOG features with validation
        """
        try:
            is_valid, message = self.validate_image_content(image)
            if not is_valid:
                raise ValueError(f"Invalid input image: {message}")

            self.logger.debug("Extracting HOG features")
            features = hog(
                image,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(3, 3),
                block_norm='L2-Hys',
                feature_vector=True,
                channel_axis=None  # Updated from multichannel=False for newer scikit-image
            )

            self.logger.debug(f"Extracted {len(features)} features")
            return features

        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            raise

    def process_image(self, image_path):
        """
        Complete image processing pipeline with enhanced validation and caching
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

    def process_images_batch(self, image_paths):
        """
        Process multiple images in parallel
        """
        try:
            # Process images in parallel using thread pool
            features = list(self.thread_pool.map(self.process_image, image_paths))
            return np.array(features)
        except Exception as e:
            self.logger.error(f"Error in batch processing: {str(e)}")
            raise