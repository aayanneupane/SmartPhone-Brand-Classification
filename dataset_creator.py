import os
import requests
from pathlib import Path
import logging
import hashlib
import time
from urllib.parse import urljoin
import trafilatura
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json

def setup_dataset_directories():
    """Create directories for each smartphone brand"""
    base_dir = Path('data')
    brands = ['iphone', 'samsung', 'pixel', 'oneplus', 'xiaomi', 'huawei']

    for brand in brands:
        (base_dir / brand).mkdir(parents=True, exist_ok=True)

    # Create checkpoint file if it doesn't exist
    checkpoint_file = base_dir / 'download_checkpoint.json'
    if not checkpoint_file.exists():
        checkpoint_file.write_text(json.dumps({}))

    return base_dir

def load_checkpoint(base_dir):
    """Load download checkpoint data"""
    checkpoint_file = base_dir / 'download_checkpoint.json'
    try:
        return json.loads(checkpoint_file.read_text())
    except Exception:
        return {}

def save_checkpoint(base_dir, checkpoint_data):
    """Save download checkpoint data"""
    checkpoint_file = base_dir / 'download_checkpoint.json'
    try:
        checkpoint_file.write_text(json.dumps(checkpoint_data))
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")

def download_image(url, save_path, headers=None, retries=3, backoff_factor=2):
    """Download an image from URL with improved retry logic"""
    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }

    filename = hashlib.md5(url.encode()).hexdigest()[:10] + '.jpg'
    file_path = save_path / filename

    # Skip if file already exists and is valid
    if file_path.exists():
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                img.verify()
            return False
        except Exception:
            file_path.unlink(missing_ok=True)

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=20)  # Increased timeout
            if response.status_code == 200 and response.headers.get('content-type', '').startswith('image/'):
                content_length = int(response.headers.get('content-length', 0))
                if content_length > 2000:  # Further reduced minimum size to 2KB
                    from PIL import Image
                    import io
                    try:
                        img = Image.open(io.BytesIO(response.content))
                        if img.size[0] > 100 and img.size[1] > 100:  # Further reduced minimum dimensions
                            img.verify()
                            img = Image.open(io.BytesIO(response.content))
                            img.save(file_path, 'JPEG', quality=85)
                            return True
                    except Exception as e:
                        logging.warning(f"Invalid image from {url}: {str(e)}")
                        if attempt < retries - 1:
                            time.sleep(backoff_factor ** attempt)
                            continue
            elif response.status_code == 429:  # Rate limit
                wait_time = min(int(response.headers.get('Retry-After', backoff_factor ** attempt)), 30)  # Cap wait time at 30 seconds
                time.sleep(wait_time)
                continue
            elif response.status_code in [403, 404]:  # Skip permanently failed URLs
                break
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(backoff_factor ** attempt)
                continue
            logging.error(f"Error downloading {url}: {str(e)}")
    return False

def extract_image_urls(html_content, base_url):
    """Extract image URLs from HTML content with improved filtering"""
    image_urls = set()
    if not html_content:
        return list(image_urls)

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    for img in soup.find_all('img'):
        for attr in ['src', 'data-src', 'data-original', 'data-lazy-src']:
            url = img.get(attr, '')
            if url:
                if url.startswith(('http://', 'https://')):
                    image_urls.add(url)
                elif not url.startswith('data:'):
                    image_urls.add(urljoin(base_url, url))

    return list(image_urls)

def download_images_batch(image_urls, save_dir, headers, checkpoint_data, brand):
    """Download a batch of images with progress tracking"""
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:  # Increased from 5 to 8 workers
        futures = []
        for url in image_urls:
            if url in checkpoint_data.get(brand, []):
                continue
            futures.append(executor.submit(download_image, url, save_dir, headers))

        for future in as_completed(futures):
            results.append(future.result())
    return sum(results)

def create_dataset():
    """Create the complete dataset with improved parallel processing and checkpointing"""
    base_dir = setup_dataset_directories()
    target_per_brand = 167  # Aim for ~1000 total images
    checkpoint_data = load_checkpoint(base_dir)

    # Additional reliable sources for each brand
    search_urls = {
        'iphone': [
            'https://www.gsmarena.com/apple-phones-48.php',
            'https://www.phonearena.com/phones/manufacturers/Apple',
            'https://www.gsmarena.com/apple_iphone_15_pro_max-12548.php',
            'https://www.gsmarena.com/apple_iphone_14_pro_max-11773.php',
            'https://www.gsmarena.com/apple_iphone_13_pro_max-11089.php',
            'https://www.gsmarena.com/apple_iphone_13-11103.php',
            'https://www.gsmarena.com/apple_iphone_12_pro_max-10237.php'
        ],
        'samsung': [
            'https://www.gsmarena.com/samsung-phones-9.php',
            'https://www.phonearena.com/phones/manufacturers/Samsung',
            'https://www.gsmarena.com/samsung_galaxy_s24_ultra-12618.php',
            'https://www.gsmarena.com/samsung_galaxy_s23_ultra-12024.php',
            'https://www.gsmarena.com/samsung_galaxy_z_fold5-12418.php',
            'https://www.gsmarena.com/samsung_galaxy_s23-12082.php',
            'https://www.gsmarena.com/samsung_galaxy_z_flip5-12252.php'
        ],
        'pixel': [
            'https://www.gsmarena.com/google-phones-107.php',
            'https://www.phonearena.com/phones/manufacturers/Google',
            'https://www.gsmarena.com/google_pixel_8_pro-12545.php',
            'https://www.gsmarena.com/google_pixel_7_pro-11908.php',
            'https://www.gsmarena.com/google_pixel_6_pro-11251.php',
            'https://www.gsmarena.com/google_pixel_8-12546.php',
            'https://www.gsmarena.com/google_pixel_7a-12170.php',
            'https://www.gsmarena.com/google_pixel_7-11903.php'
        ],
        'oneplus': [
            'https://www.gsmarena.com/oneplus-phones-95.php',
            'https://www.phonearena.com/phones/manufacturers/OnePlus',
            'https://www.gsmarena.com/oneplus_12-12615.php',
            'https://www.gsmarena.com/oneplus_11-11893.php',
            'https://www.gsmarena.com/oneplus_10_pro-11234.php'
        ],
        'xiaomi': [
            'https://www.gsmarena.com/xiaomi-phones-80.php',
            'https://www.phonearena.com/phones/manufacturers/Xiaomi',
            'https://www.gsmarena.com/xiaomi_14_pro-12551.php',
            'https://www.gsmarena.com/xiaomi_13_pro-12007.php',
            'https://www.gsmarena.com/xiaomi_12_pro-11287.php'
        ],
        'huawei': [
            'https://www.gsmarena.com/huawei-phones-58.php',
            'https://www.phonearena.com/phones/manufacturers/Huawei',
            'https://www.gsmarena.com/huawei_p60_pro-12172.php',
            'https://www.gsmarena.com/huawei_mate_60_pro+-12614.php',
            'https://www.gsmarena.com/huawei_p50_pro-10902.php'
        ]
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for brand, urls in search_urls.items():
        print(f"\nCollecting {brand} images...")
        save_dir = base_dir / brand

        # Skip if target already reached
        existing_count = len(list(save_dir.glob('*.jpg')))
        if existing_count >= target_per_brand:
            print(f"Skipping {brand}: already have {existing_count} images")
            continue

        count = existing_count
        batch_size = 20  # Process images in smaller batches

        for url in urls:
            if count >= target_per_brand:
                break

            try:
                response = requests.get(url, headers=headers, timeout=15)
                if response.status_code != 200:
                    logging.error(f"Failed to fetch {url}: Status {response.status_code}")
                    continue

                image_urls = extract_image_urls(response.text, url)
                total_batches = (len(image_urls) + batch_size - 1) // batch_size

                print(f"Found {len(image_urls)} potential images on {url}")

                with tqdm(total=min(len(image_urls), target_per_brand - count),
                         desc=f"Downloading {brand} images",
                         unit="img") as pbar:

                    for i in range(0, len(image_urls), batch_size):
                        if count >= target_per_brand:
                            break

                        batch = image_urls[i:i + batch_size]
                        successful_downloads = download_images_batch(
                            batch, save_dir, headers, checkpoint_data, brand
                        )

                        count += successful_downloads
                        pbar.update(successful_downloads)

                        # Update checkpoint
                        if brand not in checkpoint_data:
                            checkpoint_data[brand] = []
                        checkpoint_data[brand].extend(batch)
                        save_checkpoint(base_dir, checkpoint_data)

                        # Rate limiting between batches
                        time.sleep(1)

                print(f"Downloaded {count}/{target_per_brand} images for {brand}")

            except Exception as e:
                logging.error(f"Error processing {url}: {str(e)}")
                continue

            if count >= target_per_brand:
                print(f"Completed downloading {count} images for {brand}")
                break

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