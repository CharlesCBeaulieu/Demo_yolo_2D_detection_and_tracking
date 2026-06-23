import os
import urllib.request
import sys

def download_file(url, filepath):
    if os.path.exists(filepath):
        print(f"[INFO] '{filepath}' already exists. Skipping download.")
        return True
    
    print(f"[INFO] Downloading {url} to {filepath}...")
    try:
        # Custom User-Agent to avoid getting blocked by some hosting providers
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"[SUCCESS] Downloaded to '{filepath}'")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}", file=sys.stderr)
        return False

def main():
    # Create data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Files to download
    media_files = [
        {
            "url": "https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg",
            "name": "bus.jpg"
        },
        {
            "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4",
            "name": "people.mp4"
        }
    ]
    
    success = True
    for item in media_files:
        path = os.path.join(data_dir, item["name"])
        if not download_file(item["url"], path):
            success = False
            
    if success:
        print("\n[INFO] All sample assets are ready in the 'data' directory!")
    else:
        print("\n[WARNING] Some assets failed to download. Please verify your internet connection or download files manually to the 'data' directory.")

if __name__ == "__main__":
    main()
