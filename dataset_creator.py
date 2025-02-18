import os
import requests
from pathlib import Path
import logging
import hashlib
import time
from urllib.parse import urljoin
import trafilatura

def setup_dataset_directories():
    """Create directories for each smartphone brand"""
    base_dir = Path('data')
    brands = ['iphone', 'samsung', 'pixel']

    for brand in brands:
        (base_dir / brand).mkdir(parents=True, exist_ok=True)

    return base_dir

def download_image(url, save_path, headers=None):
    """Download an image from URL and save it with retries"""
    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200 and response.headers.get('content-type', '').startswith('image/'):
                # Generate unique filename based on URL
                filename = hashlib.md5(url.encode()).hexdigest()[:10] + '.jpg'
                file_path = save_path / filename

                # Check file size before saving
                content_length = int(response.headers.get('content-length', 0))
                if content_length > 5000:  # Minimum 5KB to avoid thumbnails
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    return True
                logging.warning(f"Skipping small image from {url}")
                return False

            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

        except Exception as e:
            logging.error(f"Error downloading {url}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return False

def extract_image_urls(html_content, base_url):
    """Extract image URLs from HTML content"""
    image_urls = set()
    if not html_content:
        return list(image_urls)

    # Find img tags with src attributes
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    for img in soup.find_all('img'):
        src = img.get('src', '')
        # Also check data-src for lazy-loaded images
        data_src = img.get('data-src', '')

        for url in [src, data_src]:
            if url and (url.startswith('http://') or url.startswith('https://')):
                image_urls.add(url)
            elif url and not url.startswith('data:'):  # Ignore base64 encoded images
                image_urls.add(urljoin(base_url, url))

    return list(image_urls)

def create_dataset():
    """Create the complete dataset"""
    base_dir = setup_dataset_directories()

    # Search URLs for each brand
    search_urls = {
        'iphone': [
            'https://www.gsmarena.com/apple-phones-48.php',
            'https://www.phonearena.com/phones/manufacturers/Apple'
        ],
        'samsung': [
            'https://www.gsmarena.com/samsung-phones-9.php',
            'https://www.phonearena.com/phones/manufacturers/Samsung'
        ],
        'pixel': [
            'https://www.gsmarena.com/google-phones-107.php',
            'https://www.phonearena.com/phones/manufacturers/Google'
        ]
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for brand, urls in search_urls.items():
        print(f"\nCollecting {brand} images...")
        save_dir = base_dir / brand
        count = 0

        for url in urls:
            try:
                # Fetch webpage with requests
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code != 200:
                    logging.error(f"Failed to fetch {url}: Status {response.status_code}")
                    continue

                # Extract image URLs from HTML
                image_urls = extract_image_urls(response.text, url)
                print(f"Found {len(image_urls)} potential images on {url}")

                for img_url in image_urls:
                    if count >= 50:  # Reduced from 170 to 50 images per brand
                        break

                    if download_image(img_url, save_dir, headers):
                        count += 1
                        print(f"Downloaded {count} images for {brand}")
                        time.sleep(0.5)  # Reduced from 1.5s to 0.5s

                if count >= 50:
                    print(f"Completed downloading {count} images for {brand}")
                    break

            except Exception as e:
                logging.error(f"Error processing {url}: {str(e)}")
                continue

    # Verify dataset
    total_images = sum(len(list((base_dir / brand).glob('*.jpg'))) for brand in search_urls.keys())
    print(f"\nDataset creation completed!")
    print(f"Total images downloaded: {total_images}")
    for brand in search_urls.keys():
        brand_count = len(list((base_dir / brand).glob('*.jpg')))
        print(f"{brand}: {brand_count} images")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_dataset()