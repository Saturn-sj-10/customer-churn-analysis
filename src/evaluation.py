import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)

def evaluate_model(model, X_test, y_test):
    """Evaluates model performance and prints summary metrics (Phase 9)."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}\n")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    return {
        "accuracy": acc,
        "roc_auc": roc_auc,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "fpr": fpr,
        "tpr": tpr
    }


def get_feature_importance(model, feature_names):
    """Extracts feature importances for tree models or absolute coefficients for linear models."""
    if hasattr(model, "feature_importances_"):
        importance_vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance_vals = np.abs(model.coef_[0])
    else:
        raise AttributeError("Model does not have feature_importances_ or coef_ attributes.")

    importance = pd.Series(importance_vals, index=feature_names).sort_values(ascending=False)
    return importance

def evaluate_thresholds(y_test, y_prob):
    """Part 1: Precision-Recall Threshold Curve (Phase 10)."""
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    threshold_df = pd.DataFrame({
        'Threshold': list(thresholds) + [1.0],
        'Precision': precision,
        'Recall': recall
    })
    return threshold_df


def calculate_business_impact(total_customers=100000, churn_rate=0.10, avg_revenue=900, recall=0.80, retention_success=0.25):
    """Part 4: Business Impact & Revenue Savings Calculation (Phase 10)."""
    lost_customers = total_customers * churn_rate
    potential_revenue_lost = lost_customers * avg_revenue
    
    recovered_customers = lost_customers * recall * retention_success
    recovered_revenue = recovered_customers * avg_revenue
    
    return {
        "Total Customers": total_customers,
        "Annual Lost Customers": int(lost_customers),
        "Potential Revenue Lost ($)": potential_revenue_lost,
        "Recovered Customers": int(recovered_customers),
        "Recovered Revenue ($)": recovered_revenue
    }
