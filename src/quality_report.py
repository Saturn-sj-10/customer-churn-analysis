import pandas as pd
from src.data_validation import (
    validate_schema,
    validate_dtypes,
    validate_missing,
    validate_range,
    validate_gender,
    validate_total_charges
)
from src.profiling import dataset_summary, data_quality_score
from src.utils import logger


def generate_quality_report(df):
    """
    Runs full data validation and health checks on the input DataFrame 
    and returns a structured quality summary.
    """
    logger.info("Generating Data Quality Report...")

    schema_issues = validate_schema(df)
    dtype_errors = validate_dtypes(df)
    missing_issues = validate_missing(df)
    range_issues = validate_range(df)
    gender_issues = validate_gender(df)
    charge_issues = validate_total_charges(df)
    quality_score = data_quality_score(df)
    summary = dataset_summary(df)

    report = {
        "Overall Quality Score (%)": quality_score,
        "Dataset Summary": summary,
        "Schema Issues": schema_issues,
        "Data Type Errors": dtype_errors,
        "High Missing Value Columns (>5%)": missing_issues,
        "Invalid Range Values": {k: len(v) for k, v in range_issues.items()},
        "Invalid Gender Entries Count": len(gender_issues),
        "Mismatched Total Charges Count": len(charge_issues)
    }

    logger.info(f"Data Quality Score: {quality_score}%")
    return report


if __name__ == "__main__":
    from src.data_loader import load_data
    
    # Test loading and evaluating raw data
    raw_df = load_data()
    report = generate_quality_report(raw_df)
    
    print("\n--- DATA QUALITY REPORT ---")
    for key, value in report.items():
        print(f"{key}: {value}")
