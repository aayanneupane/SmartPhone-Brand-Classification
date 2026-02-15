# Smartphone Brand Classification Dataset

A curated dataset of 832 smartphone images for brand classification.

## Dataset Structure
- Total Images: 832
- Training Set: 580 images
- Validation Set: 122 images
- Test Set: 130 images

## Classes
iphone, samsung, pixel, oneplus, xiaomi, huawei

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
