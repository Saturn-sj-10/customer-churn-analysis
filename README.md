# 📊 Enterprise Customer Churn Analysis & Retention Pipeline

## 📌 Executive Summary
This enterprise data science project analyzes customer churn across **200,000 subscriber records**, trains machine learning models to identify high-risk churners, and quantifies an estimated **~$99.4M annual revenue retention** through targeted intervention strategies.

---

## 🛠️ Tech Stack & Architecture
- **Language**: Python 3.10+
- **Data Engineering & Cleaning**: Pandas, NumPy, Custom OOP `DataCleaner`
- **Data Preprocessing**: Scikit-Learn (`StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, `Pipeline`)
- **EDA & Statistical Analysis**: Seaborn, Matplotlib, SciPy (`chi2_contingency`, `ttest_ind`)
- **Machine Learning**: `LogisticRegression`, `RandomForestClassifier`, `GridSearchCV`
- **Model Persistence**: `joblib` (`preprocessor.pkl`, `churn_model.pkl`)

---

## 📈 Model Performance & Business Impact

| Model | Churn Recall | Precision | Accuracy | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 85.0% | 62.0% | 76.0% | 0.8251 |
| **Tuned Random Forest (Final)** | **83.0%** | **73.0%** | **83.0%** | **0.8309** |

### 💰 Financial ROI Breakdown
- **Total Customer Base**: 200,000 customers
- **Annual Churn Rate**: 35.07%
- **Average Revenue / User**: $6,848.66 / year
- **Predicted At-Risk Customers**: 58,054
- **Estimated Retained Customers (25% Campaign Success)**: 14,513
- **Estimated Revenue Saved**: **$99,398,258.05**

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone [https://github.com/Saturn-sj-10/customer-churn-analysis.git](https://github.com/Saturn-sj-10/customer-churn-analysis.git)
cd customer-churn-analysis
```