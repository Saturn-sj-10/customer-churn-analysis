import pandas as pd
from config import RAW_DATA
from src.utils import logger

def load_data():
    """Loads raw Excel dataset into a Pandas DataFrame."""
    try:
        logger.info(f"Loading raw data from: {RAW_DATA}")
        df = pd.read_excel(RAW_DATA)
        logger.info(f"Successfully loaded dataset with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading raw data from {RAW_DATA}: {e}")
        raise  # Preserves full original traceback cleanly
