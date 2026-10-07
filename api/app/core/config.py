"""Конфигурация API через переменные окружения."""

import os


class Settings:
    def __init__(self) -> None:
        self.app_name: str = os.getenv("APP_NAME", "Dora B2BCase API")
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
        # catalog | mock | qdrant
        self.classifier_backend: str = os.getenv("CLASSIFIER_BACKEND", "catalog").lower()
        self.qdrant_url: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
        # Явный путь к каталогу; если пусто — резолвер ищет DATA_DIR/data.csv
        self.catalog_path: str = os.getenv("CATALOG_PATH", "") or ""
        self.data_dir: str = os.getenv("DATA_DIR", "data")


settings = Settings()
