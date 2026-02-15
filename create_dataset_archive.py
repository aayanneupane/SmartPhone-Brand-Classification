import os
import zipfile
from pathlib import Path
import logging

def create_dataset_archive():
    """Create a zip archive of the dataset"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize counters
    total_images = 0
    brands = ['iphone', 'samsung', 'pixel', 'oneplus', 'xiaomi', 'huawei']
    
    # Create zip file
    with zipfile.ZipFile('smartphone_dataset.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for brand in brands:
            brand_dir = Path('data') / brand
            if brand_dir.exists():
                # Count images
                images = list(brand_dir.glob('*.jpg'))
                total_images += len(images)
                
                # Add images to zip
                logger.info(f"Adding {len(images)} {brand} images to archive...")
                for img in images:
                    zipf.write(img, img.relative_to(Path('data')))
    
    # Get archive size
    archive_size = Path('smartphone_dataset.zip').stat().st_size / (1024 * 1024)  # Convert to MB
    
    logger.info(f"\nArchive created successfully!")
    logger.info(f"Total images: {total_images}")
    logger.info(f"Archive size: {archive_size:.1f} MB")
    logger.info("You can now download 'smartphone_dataset.zip'")

if __name__ == "__main__":
    create_dataset_archive()
