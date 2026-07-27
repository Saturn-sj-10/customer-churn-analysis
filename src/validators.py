import pandas as pd

EXPECTED_COLUMNS = [
    "customer_id",
    "full_name",
    "gender",
    "city",
    "contract_type",
    "subscription_type",
    "internet_service",
    "monthly_charges",
    "total_charges",
    "tenure_months",
    "support_tickets",
    "complaint_count",
    "avg_monthly_usage_gb",
    "churn"
]

def validate_schema(df):

    missing_columns = [
        col
        for col in EXPECTED_COLUMNS
        if col not in df.columns
    ]

    extra_columns = [
        col
        for col in df.columns
        if col not in EXPECTED_COLUMNS
    ]

    return {
        "Missing Columns": missing_columns,
        "Unexpected Columns": extra_columns
    }

EXPECTED_DTYPES = {
    "monthly_charges": "float64",
    "tenure_months": "int64",
    "gender": "object"
}

def validate_dtypes(df):

    errors = []

    for col, dtype in EXPECTED_DTYPES.items():

        if col in df.columns:

            if str(df[col].dtype) != dtype:

                errors.append(
                    {
                        "Column": col,
                        "Expected": dtype,
                        "Found": str(df[col].dtype)
                    }
                )

    return errors


def validate_missing(df, threshold=5):

    report = []

    for col in df.columns:

        pct = (
            df[col].isnull().mean()
            *100
        )

        if pct > threshold:

            report.append({
                "Column": col,
                "Missing %": round(pct,2)
            })

    return report


def validate_gender(df):

    allowed = {
        "Male",
        "Female"
    }

    invalid = df[
        ~df["gender"].isin(allowed)
    ]

    return invalid

def validate_range(df):
    """Validates that numerical features do not contain negative values."""
    issues = {}

    if "monthly_charges" in df.columns:
        # Convert to numeric safely, treating string errors as NaN
        monthly_num = pd.to_numeric(df["monthly_charges"], errors="coerce")
        issues["Negative Charges"] = len(df[monthly_num < 0])

    if "tenure_months" in df.columns:
        tenure_num = pd.to_numeric(df["tenure_months"], errors="coerce")
        issues["Negative Tenure"] = len(df[tenure_num < 0])

    if "total_charges" in df.columns:
        total_num = pd.to_numeric(df["total_charges"], errors="coerce")
        issues["Negative Total Charges"] = len(df[total_num < 0])

    return issues

def validate_phone(df):
    if "phone" not in df.columns:
        return df.iloc[0:0]

    return df[
        ~df["phone"]
        .astype(str)
        .str.match(r"^\d{10}$", na=False)
    ]

def validate_dates(df):
    if "customer_since" not in df.columns:
        return {}

    customer_since = pd.to_datetime(
        df["customer_since"],
        errors="coerce"
    )

    today = pd.Timestamp.today().normalize()

    issues = {
        "Invalid or Future Customer Dates": df[
            customer_since.isna()
            | (customer_since > today)
        ]
    }

    if "date_of_birth" in df.columns:
        date_of_birth = pd.to_datetime(
            df["date_of_birth"],
            errors="coerce"
        )

        issues["Customer Before Birth Date"] = df[
            customer_since < date_of_birth
        ]

    return issues


def validate_total_charges(df):
    """Checks for significant discrepancies between total_charges and expected total (monthly_charges * tenure_months)."""
    if not all(col in df.columns for col in ["monthly_charges", "tenure_months", "total_charges"]):
        return pd.DataFrame()

    # Convert columns to float safely (invalid string entries become NaN)
    monthly = pd.to_numeric(df["monthly_charges"], errors="coerce")
    tenure = pd.to_numeric(df["tenure_months"], errors="coerce")
    total = pd.to_numeric(df["total_charges"], errors="coerce")

    # Calculate expected total charges
    expected = monthly * tenure
    
    # Filter rows with valid non-null numeric values and a discrepancy > $100
    valid_mask = monthly.notnull() & tenure.notnull() & total.notnull()
    diff = (expected - total).abs()
    
    invalid = df[valid_mask & (diff > 100)]
    return invalid

def validate_login(df):
    if "last_login_date" not in df.columns:
        return df.iloc[0:0]

    return df[
        (df["churn"] == "Yes")
        &
        (df["last_login_date"] > "2026-07-01")
    ]


