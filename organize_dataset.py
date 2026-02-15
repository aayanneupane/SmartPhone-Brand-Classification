import os
import shutil
import json
from pathlib import Path
import random
from PIL import Image
import logging
import zipfile

def setup_logger():
    """Configure logging for the dataset organization process"""
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def create_dataset_structure():
    """Create the directory structure for the organized dataset"""
    base_dir = Path('organized_dataset')
    splits = ['train', 'val', 'test']

    # Remove if exists and create new
    if base_dir.exists():
        shutil.rmtree(base_dir)

    # Create split directories
    for split in splits:
        for brand in ['iphone', 'samsung', 'pixel', 'oneplus', 'xiaomi', 'huawei']:
            (base_dir / split / brand).mkdir(parents=True, exist_ok=True)

    return base_dir

def get_image_metadata(image_path):
    """Extract metadata from image"""
    try:
        with Image.open(image_path) as img:
            return {
                'width': img.size[0],
                'height': img.size[1],
                'mode': img.mode,
                'format': img.format
            }
    except Exception as e:
        return None

def organize_dataset():
    """Organize the dataset into train/val/test splits with metadata"""
    logger = setup_logger()
    base_dir = create_dataset_structure()

    # Dataset statistics
    stats = {
        'total_images': 0,
        'splits': {'train': 0, 'val': 0, 'test': 0},
        'brands': {}
    }

    # Metadata for each image
    metadata = {
        'dataset_info': {
            'name': 'Smartphone Brand Classification Dataset',
            'version': '1.0',
            'description': 'A curated dataset of smartphone images for brand classification',
            'classes': ['iphone', 'samsung', 'pixel', 'oneplus', 'xiaomi', 'huawei']
        },
        'images': {}
    }

    source_dir = Path('data')
    for brand in metadata['dataset_info']['classes']:
        brand_dir = source_dir / brand
        if not brand_dir.exists():
            logger.warning(f"Brand directory not found: {brand}")
            continue

        # Get all images for this brand
        images = list(brand_dir.glob('*.jpg'))
        random.shuffle(images)

        # Calculate split sizes
        n_images = len(images)
        n_train = int(0.7 * n_images)
        n_val = int(0.15 * n_images)

        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }

        stats['brands'][brand] = {
            'total': n_images,
            'splits': {split: len(imgs) for split, imgs in splits.items()}
        }

        # Copy images to new structure
        for split_name, split_images in splits.items():
            for img_path in split_images:
                # Create a standardized filename
                new_filename = f"{brand}_{img_path.stem}.jpg"
                new_path = base_dir / split_name / brand / new_filename

                # Copy image
                shutil.copy2(img_path, new_path)

                # Get and store metadata
                img_metadata = get_image_metadata(img_path)
                if img_metadata:
                    metadata['images'][new_filename] = {
                        'split': split_name,
                        'brand': brand,
                        'metadata': img_metadata
                    }

                stats['splits'][split_name] += 1
                stats['total_images'] += 1

    # Save metadata and stats
    with open(base_dir / 'dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    with open(base_dir / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # Create README
    readme_content = f"""# Smartphone Brand Classification Dataset

A curated dataset of {stats['total_images']} smartphone images for brand classification.

## Dataset Structure
- Total Images: {stats['total_images']}
- Training Set: {stats['splits']['train']} images
- Validation Set: {stats['splits']['val']} images
- Test Set: {stats['splits']['test']} images

## Classes
{', '.join(metadata['dataset_info']['classes'])}

## Directory Structure
- organized_dataset/
  - train/
    - iphone/
    - samsung/
    - pixel/
    - oneplus/
    - xiaomi/
    - huawei/
  - val/
    - (same structure as train)
  - test/
    - (same structure as train)
  - dataset_metadata.json
  - dataset_stats.json
  - README.md

## Usage
1. The dataset is split into train/validation/test sets (70/15/15 split)
2. Each image filename follows the format: brand_uniqueid.jpg
3. Metadata for each image is available in dataset_metadata.json
4. Overall statistics can be found in dataset_stats.json

## License
This dataset is for educational and research purposes only.
"""

    with open(base_dir / 'README.md', 'w') as f:
        f.write(readme_content)

    # Create new archive
    logger.info("Creating new archive...")
    archive_name = 'organized_smartphone_dataset.zip'
    with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in base_dir.rglob('*'):
            if file.is_file():
                zipf.write(file, file.relative_to(base_dir))

    # Log completion
    logger.info("\nDataset Organization Summary:")
    logger.info(f"Total images processed: {stats['total_images']}")
    for split, count in stats['splits'].items():
        logger.info(f"{split.capitalize()} set: {count} images")
    logger.info("\nBrand-wise distribution:")
    for brand, brand_stats in stats['brands'].items():
        logger.info(f"{brand}: {brand_stats['total']} images")
    logger.info(f"\nOrganized dataset saved as '{archive_name}'")

if __name__ == "__main__":
    organize_dataset()