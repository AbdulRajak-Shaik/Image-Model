import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from PIL import Image

class DatasetDownloader:
    def __init__(self, root_dir="datasets"):
        self.root_dir = Path(root_dir)
        self.dataset_dirs = {
            "faceforensics++": self.root_dir / "faceforensics++",
            "celeb_df_v2": self.root_dir / "celeb_df_v2",
            "dfdc": self.root_dir / "dfdc",
            "deeperforensics": self.root_dir / "deeperforensics",
            "google_dfd": self.root_dir / "google_dfd"
        }

    def setup_directories(self):
        """Creates the dataset folders and subfolders for train, val, and test splits."""
        for name, path in self.dataset_dirs.items():
            for split in ["train", "val", "test"]:
                for label in ["Real", "Fake"]:
                    os.makedirs(path / split / label, exist_ok=True)
        print(f"Directory structure initialized under '{self.root_dir}/'")

    def download_celeb_df_v2_sample(self):
        """Downloads a small sample set of Celeb-DF v2 for testing/dev if URL is accessible."""
        # Note: True Celeb-DF v2 dataset requires request, but we download a public sample mock zip for quick-start.
        sample_url = "https://github.com/ondyari/FaceForensics/releases/download/v1.0/faceforensics_benchmarks.zip" # Placeholder fallback
        dest_zip = self.root_dir / "celeb_df_sample.zip"
        
        try:
            print(f"Downloading Celeb-DF sample from {sample_url}...")
            urllib.request.urlretrieve(sample_url, dest_zip)
            with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
                zip_ref.extractall(self.dataset_dirs["celeb_df_v2"] / "train")
            os.remove(dest_zip)
            print("Celeb-DF sample extracted successfully.")
        except Exception as e:
            print(f"Automatic download failed (Normal for restricted research datasets): {e}")
            self.print_manual_instructions("celeb_df_v2")

    def print_manual_instructions(self, dataset_name):
        instructions = {
            "faceforensics++": (
                "1. Request access from the official FaceForensics++ GitHub repository.\n"
                "2. Run their python downloader script `faceforensics_download_v4.py`.\n"
                "3. Extract and place frames/videos into 'datasets/faceforensics++/<split>/[Real|Fake]'."
            ),
            "celeb_df_v2": (
                "1. Request Celeb-DF v2 by contacting the authors via: https://github.com/yuezunli/celeb-deepfakeforensics\n"
                "2. Download both Celeb-real and Celeb-synthesis videos.\n"
                "3. Place or extract under 'datasets/celeb_df_v2/train/Real' and 'datasets/celeb_df_v2/train/Fake' respectively."
            ),
            "dfdc": (
                "1. Navigate to the Deepfake Detection Challenge website on Kaggle: https://www.kaggle.com/c/deepfake-detection-challenge\n"
                "2. Download the dataset zip archives.\n"
                "3. Extract to 'datasets/dfdc/' maintaining the train/val/test splits."
            ),
            "deeperforensics": (
                "1. Request DeeperForensics-1.0 via their website: https://github.com/EndlessSly/DeeperForensics-1.0\n"
                "2. Download the source videos and structure into 'datasets/deeperforensics/'."
            ),
            "google_dfd": (
                "1. Access the Google DFD dataset from the FaceForensics++ download agreement.\n"
                "2. Download raw and compressed videos, placing them into 'datasets/google_dfd/'."
            )
        }
        
        print("\n" + "="*50)
        print(f"MANUAL SETUP INSTRUCTIONS FOR {dataset_name.upper()}:")
        print("="*50)
        print(instructions.get(dataset_name, "No instructions available."))
        print("="*50 + "\n")

    def sanitize_dataset(self, dataset_name):
        """Scans the dataset directories, handles corrupted files, and logs skipped files."""
        path = self.dataset_dirs.get(dataset_name)
        if not path or not path.exists():
            print(f"Dataset path {path} does not exist. Skipping sanitation.")
            return

        print(f"Sanitizing {dataset_name} dataset...")
        corrupt_count = 0
        
        for file_path in path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                try:
                    with Image.open(file_path) as img:
                        img.verify() # Verify image integrity
                except (IOError, SyntaxError) as e:
                    print(f"[CORRUPT] Removing and logging corrupt image: {file_path}. Error: {e}")
                    os.remove(file_path)
                    corrupt_count += 1
                    
        print(f"Sanitation complete. Removed {corrupt_count} corrupt files.")

if __name__ == "__main__":
    downloader = DatasetDownloader()
    downloader.setup_directories()
    # Requesting Celeb-DF manual download instructions
    downloader.print_manual_instructions("celeb_df_v2")
