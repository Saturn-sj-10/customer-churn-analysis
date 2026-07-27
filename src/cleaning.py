import numpy as np
import pandas as pd

from src.utils import logger


DATE_COLUMNS = [
    "customer_since",
    "date_of_birth",
    "last_login_date",
    "last_payment_date"
]


class DataCleaner:

    def __init__(self, df):
        self.df = df.copy()

    def remove_duplicates(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = before - len(self.df)

        logger.info(f"Removed {removed} duplicate rows.")
        return self

    def clean_gender(self):
        if "gender" not in self.df.columns:
            return self

        mapping = {
            "male": "Male",
            "MALE": "Male",
            "M": "Male",
            "female": "Female",
            "F": "Female"
        }

        self.df["gender"] = self.df["gender"].astype(str).str.strip().replace(mapping)

        logger.info("Gender standardized.")
        return self

    def clean_city(self):
        if "city" not in self.df.columns:
            return self

        city_mapping = {
            "DELHI": "Delhi",
            "delhi": "Delhi",
            "New Delhi": "Delhi",
            "N. Delhi": "Delhi"
        }

        self.df["city"] = self.df["city"].replace(city_mapping)

        logger.info("City names standardized.")
        return self

    def clean_monthly_charges(self):
        if "monthly_charges" not in self.df.columns:
            return self

        self.df["monthly_charges"] = (
            self.df["monthly_charges"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.strip()
        )

        self.df["monthly_charges"] = pd.to_numeric(
            self.df["monthly_charges"],
            errors="coerce"
        )

        logger.info("Monthly charges converted.")
        return self

    def clean_dates(self):
        for col in DATE_COLUMNS:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(
                    self.df[col],
                    errors="coerce"
                )

        logger.info("Date columns converted.")
        return self

    def clean_negative_values(self):
        """Replaces or filters negative values in charges and tenure."""
        # Handle negative monthly charges
        if "monthly_charges" in self.df.columns:
            self.df["monthly_charges"] = self.df["monthly_charges"].apply(lambda x: abs(x) if pd.notnull(x) else x)
    
        # Handle negative total charges
        if "total_charges" in self.df.columns:
            self.df["total_charges"] = self.df["total_charges"].apply(lambda x: abs(x) if pd.notnull(x) else x)
        
        # Handle negative tenure
        if "tenure_months" in self.df.columns:
            self.df["tenure_months"] = self.df["tenure_months"].apply(lambda x: abs(x) if pd.notnull(x) else x)

        return self

    def fill_missing(self):
        """Imputes missing values and ensures correct data types."""
        # Impute numeric features with median
        numeric_cols = self.df.select_dtypes(include=["float64", "int64"]).columns
        for col in numeric_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())

        # Cast tenure back to integer
        if "tenure_months" in self.df.columns:
            self.df["tenure_months"] = self.df["tenure_months"].astype("int64")

        return self

    def verify(self):
        logger.info(f"Rows: {len(self.df)}")
        logger.info(
            f"Duplicates: {self.df.duplicated().sum()}"
        )
        logger.info(
            f"Missing Values:\n{self.df.isnull().sum()}"
        )

        return self

    def save(self, path):
        self.df.to_csv(path, index=False)

        logger.info("Processed dataset saved.")
        return self
