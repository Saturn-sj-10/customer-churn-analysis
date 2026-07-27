import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import shapiro, sem, chi2_contingency, ttest_ind, f_oneway


def descriptive_statistics(df):
    """Returns transposed descriptive statistics for numerical features."""
    return df.describe().T


def plot_distribution(df, column="monthly_charges"):
    """Plots histogram and Q-Q plot for normality visual checks."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(df[column].dropna(), bins=30, edgecolor="black")
    axes[0].set_title(f"{column.replace('_', ' ').title()} Distribution")
    
    # Q-Q Plot
    stats.probplot(df[column].dropna(), dist="norm", plot=axes[1])
    axes[1].set_title(f"Q-Q Plot for {column.replace('_', ' ').title()}")
    
    plt.tight_layout()
    plt.show()


def test_normality(df, column="monthly_charges"):
    """Performs Shapiro-Wilk test for normality on a sample if size > 5000."""
    data = df[column].dropna()
    if len(data) > 5000:
        data = data.sample(5000, random_state=42)
    
    stat, p = shapiro(data)
    return {"Statistic": round(stat, 4), "p-value": p, "Is Normal": p > 0.05}


def calculate_confidence_interval(df, column="monthly_charges", confidence=0.95):
    """Calculates Mean and Confidence Interval using Standard Error of Mean (SEM)."""
    data = df[column].dropna()
    mean = data.mean()
    ci = sem(data) * 1.96  # 95% CI approximation
    return {
        "Mean": round(mean, 2),
        "Lower CI": round(mean - ci, 2),
        "Upper CI": round(mean + ci, 2)
    }


def test_chi2_churn(df, cat_col="contract_type", target_col="churn"):
    """Chi-Square Test of Independence between a categorical variable and churn."""
    table = pd.crosstab(df[cat_col], df[target_col])
    chi2, p, dof, expected = chi2_contingency(table)
    return {"Chi2 Statistic": round(chi2, 4), "p-value": p, "Degrees of Freedom": dof}


def test_ttest_churn(df, num_col="monthly_charges", target_col="churn"):
    """Welch's t-test comparing a numerical feature between churn groups."""
    churn_yes = df[df[target_col] == "Yes"][num_col].dropna()
    churn_no = df[df[target_col] == "No"][num_col].dropna()
    
    t, p = ttest_ind(churn_yes, churn_no, equal_var=False)
    return {"t-statistic": round(t, 4), "p-value": p}


def test_anova_contract(df, num_col="monthly_charges", cat_col="contract_type"):
    """One-way ANOVA across contract types."""
    groups = [group[num_col].dropna() for name, group in df.groupby(cat_col)]
    f, p = f_oneway(*groups)
    return {"F-statistic": round(f, 4), "p-value": p}


def compute_correlation(df, cols=None):
    """Computes correlation matrix for specified numerical columns."""
    if cols is None:
        cols = ["tenure_months", "total_charges"]
    return df[cols].corr().round(4)
