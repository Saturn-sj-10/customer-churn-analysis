import pandas as pd

def dataset_summary(df):
    """Returns basic shape and memory usage information."""
    return {
        'Rows': df.shape[0],
        'Columns': df.shape[1],
        'Memory Usage(MB)': round(
            df.memory_usage(deep=True).sum() / (1024**2),
            2
        )
    }


def datatype_report(df):
    """Returns data types of all columns."""
    report = df.dtypes.reset_index()
    report.columns = ["Column", "Data Type"]
    return report


def missing_report(df):
    """Calculates missing count and missing percentage per column."""
    report = df.isnull().sum().to_frame("Missing Count")
    report["Missing %"] = (report["Missing Count"] / len(df) * 100).round(2)
    return report.sort_values(by="Missing %", ascending=False)


def duplicate_report(df):
    """Calculates total duplicate rows and duplicate percentage."""
    duplicate_count = df.duplicated().sum()
    duplicate_percentage = round(duplicate_count / len(df) * 100, 2) if len(df) > 0 else 0.0
    return {
        "Duplicate Count": duplicate_count,
        "Duplicate %": duplicate_percentage
    }


def unique_report(df):
    """Calculates number of unique values per column."""
    report = pd.DataFrame({"Unique Values": df.nunique()})
    return report.sort_values(by="Unique Values", ascending=False)


def numerical_summary(df):
    """Generates descriptive statistics for numerical columns."""
    num_df = df.select_dtypes(include=['number'])
    if num_df.empty:
        return pd.DataFrame()
    return num_df.describe().T


def categorical_summary(df):
    """Generates descriptive statistics for categorical/object columns safely."""
    cat_df = df.select_dtypes(include=['object', 'category', 'string'])
    if cat_df.empty:
        return pd.DataFrame(columns=['count', 'unique', 'top', 'freq'])
    return cat_df.describe().T


def data_quality_score(df):
    """Computes an overall data health quality score (0 - 100%)."""
    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()
    total_cells = df.shape[0] * df.shape[1]
    
    if total_cells == 0:
        return 100.0

    quality = (1 - ((missing + duplicates) / total_cells)) * 100
    return round(quality, 2)


def profile_dataset(df):
    """Runs full automated data profiling suite on the DataFrame."""
    return {
        "Summary": dataset_summary(df),
        "Data Types": datatype_report(df),
        "Missing": missing_report(df),
        "Duplicates": duplicate_report(df),
        "Unique": unique_report(df),
        "Numerical": numerical_summary(df),
        "Categorical": categorical_summary(df),
        "Quality Score": data_quality_score(df)
    }
