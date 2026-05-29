import requests
import zipfile
import io
from pathlib import Path
from .base import BaseDataset

DATASETS = {
    "Apache": "https://zenodo.org/record/8275861/files/Apache.zip?download=1",
    "HDFS": "https://zenodo.org/record/8275861/files/HDFS.zip?download=1",
    "BGL": "https://zenodo.org/record/8275861/files/BGL.zip?download=1",
    "Hadoop": "https://zenodo.org/record/8275861/files/Hadoop.zip?download=1",
    "HPC": "https://zenodo.org/record/8275861/files/HPC.zip?download=1",
    "Linux": "https://zenodo.org/record/8275861/files/Linux.zip?download=1",
    "Mac": "https://zenodo.org/record/8275861/files/Mac.zip?download=1",
    "OpenSSH": "https://zenodo.org/record/8275861/files/OpenSSH.zip?download=1",
    "OpenStack": "https://zenodo.org/record/8275861/files/OpenStack.zip?download=1",
    "Proxifier": "https://zenodo.org/record/8275861/files/Proxifier.zip?download=1",
    "Spark": "https://zenodo.org/record/8275861/files/Spark.zip?download=1",
    "Thunderbird": "https://zenodo.org/record/8275861/files/Thunderbird.zip?download=1",
    "Zookeeper": "https://zenodo.org/record/8275861/files/Zookeeper.zip?download=1",
}

class LoghubDataset(BaseDataset):
    def __init__(self, name: str, data_dir: str = "data"):
        if name not in DATASETS:
            raise ValueError(f"Dataset {name} not supported. Choose from {list(DATASETS.keys())}")
        
        self.name = name
        self.url = DATASETS[name]
        self.data_dir = Path(data_dir)
        self.dataset_dir = self.data_dir / name

    def download(self, force=False, debug=False) -> None:
        if self.dataset_dir.exists() and not force:
            if debug:
                print(f"Dataset {self.name} already exists at {self.dataset_dir}")
            return

        if debug:
            print(f"Downloading {self.name} dataset from {self.url}...")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        response = requests.get(self.url, stream=True)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            if debug:
                print(f"Extracting {self.name}...")
            z.extractall(self.data_dir)

        if debug:
            print(f"{self.name} dataset ready.")

    def get_log_path(self) -> Path:
        log_file = self.dataset_dir / f"{self.name}.log"
        if not log_file.exists():
            log_files = list(self.dataset_dir.glob("*.log"))
            if log_files:
                return log_files[0]
            raise FileNotFoundError(f"Could not find log file in {self.dataset_dir}")
        return log_file
