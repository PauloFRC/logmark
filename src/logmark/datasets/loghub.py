import os
import requests
import zipfile
import io
from pathlib import Path
from .base import BaseDataset

DATASETS = {
    "HDFS": "https://zenodo.org/record/8275861/files/HDFS.zip?download=1",
    "BGL": "https://zenodo.org/record/8275861/files/BGL.zip?download=1",
}

class LoghubDataset(BaseDataset):
    def __init__(self, name: str, data_dir: str = "data"):
        if name not in DATASETS:
            raise ValueError(f"Dataset {name} not supported. Choose from {list(DATASETS.keys())}")
        
        self.name = name
        self.url = DATASETS[name]
        self.data_dir = Path(data_dir)
        self.dataset_dir = self.data_dir / name

    def download(self, force: bool = False) -> None:
        if self.dataset_dir.exists() and not force:
            print(f"Dataset {self.name} already exists at {self.dataset_dir}")
            return

        print(f"Downloading {self.name} dataset from {self.url}...")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        response = requests.get(self.url, stream=True)
        response.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            print(f"Extracting {self.name}...")
            z.extractall(self.data_dir)
        
        print(f"{self.name} dataset ready.")

    def get_log_path(self) -> Path:
        log_file = self.dataset_dir / f"{self.name}.log"
        if not log_file.exists():
            log_files = list(self.dataset_dir.glob("*.log"))
            if log_files:
                return log_files[0]
            raise FileNotFoundError(f"Could not find log file in {self.dataset_dir}")
        return log_file
