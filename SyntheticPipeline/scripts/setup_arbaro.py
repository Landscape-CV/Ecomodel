import subprocess
import os
from pathlib import Path

ARBARO_URL = "https://downloads.sourceforge.net/project/arbaro/arbaro/1.9.8/arbaro_1_9_8.zip"
DEST_DIR = Path(__file__).parent.parent / "lib" / "arbaro"
ZIP_PATH = DEST_DIR / "arbaro.zip"

def download_and_extract():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading Arbaro from {ARBARO_URL}...")
    subprocess.run(["curl.exe", "-L", ARBARO_URL, "-o", str(ZIP_PATH)], check=True)
        
    print(f"Extracting to {DEST_DIR}...")
    subprocess.run(["powershell", "-Command", f"Expand-Archive -Path '{ZIP_PATH}' -DestinationPath '{DEST_DIR}' -Force"], check=True)
        
    # Clean up the zip file
    os.remove(ZIP_PATH)
    
    # Verify
    jar_path = DEST_DIR / "arbaro_cmd.jar"
    if jar_path.exists():
        print(f"Arbaro successfully installed at: {jar_path}")
    else:
        print("Installation finished, but couldn't locate arbaro_cmd.jar. Check the directory structure.")

if __name__ == "__main__":
    download_and_extract()
