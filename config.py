from pathlib import Path

# Dynamically finds the project root directory
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA = DATA_DIR / "raw" / "Customer_Churn_Enterprise_Dataset.xlsx"  # Fixed filename
PROCESSED_DATA = DATA_DIR / "processed"
REPORTS = BASE_DIR / "reports"
IMAGES = BASE_DIR / "images"
LOGS = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"

# Automatically create missing output folders
for folder in [PROCESSED_DATA, REPORTS, IMAGES, LOGS, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
