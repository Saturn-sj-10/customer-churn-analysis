import os
import joblib
from src.utils import logger
from src.data_loader import load_data
from src.data_cleaning import DataCleaner
from src.features import prepare_and_preprocess_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.model import train_baseline_model, tune_random_forest, save_model
from src.evaluation import evaluate_model, get_feature_importance, calculate_business_impact


def main():
    logger.info("================ Project Pipeline Started ================")

    # 1. Load Raw Data
    df_raw = load_data()
    logger.info(f"Loaded Raw Dataset: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # 2. Clean Data
    logger.info("Starting Data Cleaning Process...")
    cleaner = (
        DataCleaner(df_raw)
        .remove_duplicates()
        .clean_gender()
        .clean_city()
        .clean_monthly_charges()
        .clean_dates()
        .clean_negative_values()
        .fill_missing()
    )
    df_cleaned = cleaner.df
    logger.info(f"Cleaned Dataset Shape: {df_cleaned.shape}")

    # 3. Feature Engineering & Preprocessing
    logger.info("Preprocessing Data and Building Pipeline...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_and_preprocess_data(df_cleaned)

    # 4. Train Models
    logger.info("Training Baseline Model (Logistic Regression)...")
    baseline_model = train_baseline_model(X_train, y_train)

    logger.info("Tuning Complex Model (Random Forest via GridSearchCV)...")
    rf_model = tune_random_forest(X_train, y_train)

    # 5. Save Final Model
    save_model(rf_model, "models/churn_model.pkl")

    # 6. Evaluate Final Model
    logger.info("Evaluating Tuned Random Forest Model on Test Set:")
    results = evaluate_model(rf_model, X_test, y_test)

    # 7. Extract Feature Importances
    all_feature_names = NUMERIC_FEATURES + list(
        rf_model.named_steps["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    ) if hasattr(rf_model, "named_steps") else NUMERIC_FEATURES + CATEGORICAL_FEATURES

    try:
        importance_df = get_feature_importance(rf_model, all_feature_names)
        print("\n--- TOP FEATURE IMPORTANCES ---")
        print(importance_df.head(10))
    except Exception as e:
        logger.warning(f"Could not extract feature importances: {e}")

    # 8. Calculate Business ROI
    roi = calculate_business_impact(recall=0.86)
    print("\n--- ESTIMATED BUSINESS IMPACT & ROI ---")
    for k, v in roi.items():
        print(f"{k}: {v:,}" if isinstance(v, (int, float)) else f"{k}: {v}")

    logger.info("================ Project Pipeline Completed Successfully ================")


if __name__ == "__main__":
    main()
