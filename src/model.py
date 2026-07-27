import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_baseline_model(X_train, y_train):
    """Trains a baseline Logistic Regression model."""
    model = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)
    return model

def tune_random_forest(X_train, y_train):
    """Tunes a Random Forest Classifier using GridSearchCV prioritized for Recall."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None]
    }
    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, class_weight="balanced"),
        param_grid=param_grid,
        scoring="recall",
        cv=5,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def save_model(model, filepath="models/churn_model.pkl"):
    """Saves the trained model to disk safely."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)  # Auto-creates 'models/' if missing
    joblib.dump(model, path)

def load_model(filepath="models/churn_model.pkl"):
    """Loads a trained model from disk."""
    return joblib.load(filepath)
