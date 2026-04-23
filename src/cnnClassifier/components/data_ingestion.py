import os
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.data_path = Path(self.config.unzip_dir)

    def download_file(self) -> None:
        """
        Skip downloading. Ensure dataset already exists locally.
        """
        if self._is_data_available():
            logger.info(f"✅ Using existing dataset at: {self.data_path.resolve()}")
        else:
            raise FileNotFoundError(
                f"❌ Dataset not found at {self.data_path.resolve()}.\n"
                "Please place your dataset in the correct structure:\n"
                "artifacts/data_ingestion/train, val, test"
            )

    def extract_zip_file(self) -> None:
        """
        Skip extraction since dataset is already organized.
        """
        logger.info("✅ Skipping unzip step (using pre-existing dataset).")

    def _is_data_available(self) -> bool:
        """
        Check if train, val, and test folders exist with class subfolders.
        """
        required_dirs = ["train", "val", "test"]

        for d in required_dirs:
            dir_path = self.data_path / d
            if not dir_path.exists() or not any(dir_path.iterdir()):
                logger.warning(f"⚠️ Missing or empty folder: {dir_path}")
                return False

        logger.info("✅ Dataset structure verified (train/val/test found).")
        return True