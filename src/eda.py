import pandas as pd

def churn_rate(df):

    churn = (

        df["churn"]

        .value_counts(normalize=True)

        *100

    )

    return churn.round(2)

def contract_churn(df):

    return (

        pd.crosstab(

            df["contract_type"],

            df["churn"],

            normalize="index"

        )*100

    ).round(2)

def tenure_churn(df):

    return df.groupby(

        "churn"

    )["tenure_months"].mean()

def complaint_analysis(df):

    return (

        df.groupby(

            "churn"

        )["complaint_count"]

        .mean()

    )

def monthly_charge_analysis(df):

    return (

        df.groupby(

            "churn"

        )["monthly_charges"]

        .mean()

    )

def city_churn(df):

    return (

        pd.crosstab(

            df["city"],

            df["churn"],

            normalize="index"

        )*100

    ).round(2)

def internet_analysis(df):

    return (

        pd.crosstab(

            df["internet_service"],

            df["churn"],

            normalize="index"

        )*100

    ).round(2)


