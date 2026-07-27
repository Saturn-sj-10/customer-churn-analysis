{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e80bed08-e3dd-419b-a2e6-24fdb25754c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 📊 Customer Churn Analysis & ML Retention Pipeline\n",
    "\n",
    "## 📌 Executive Summary\n",
    "This enterprise data science project predicts customer churn across **200,000 subscriber records**, identifying high-risk churners and quantifying a potential **~$99.4M annual revenue retention** through targeted intervention strategies.\n",
    "\n",
    "## 🛠️ Architecture & Tech Stack\n",
    "- **Data Engineering & Cleaning**: Pandas, NumPy, Scikit-Learn (Custom OOP `DataCleaner`)\n",
    "- **EDA & Statistical Testing**: Seaborn, Matplotlib, SciPy (`chi2_contingency`, `ttest_ind`)\n",
    "- **Machine Learning**: Scikit-Learn (`StandardScaler`, `OneHotEncoder`, `RandomForestClassifier`, `GridSearchCV`)\n",
    "- **Pipeline Persistence**: `joblib` (`preprocessor.pkl`, `churn_model.pkl`)\n",
    "\n",
    "## 📈 Key Results & Impact\n",
    "| Metric | Baseline (Logistic Regression) | Tuned Random Forest |\n",
    "| :--- | :--- | :--- |\n",
    "| **Accuracy** | 76.0% | **83.0%** |\n",
    "| **Churn Recall** | 85.0% | **83.0%** |\n",
    "| **Churn Precision** | 62.0% | **73.0% (+11%)** |\n",
    "| **ROC-AUC** | 0.8251 | **0.8309** |\n",
    "\n",
    "- **At-Risk Customers Identified**: 58,054\n",
    "- **Estimated Customers Retained**: 14,513\n",
    "- **Financial Impact**: **$99,398,258.05 Retained Revenue**\n",
    "\n",
    "## 🚀 How to Run\n",
    "```bash\n",
    "# Clone repository\n",
    "git clone [https://github.com/your-username/customer-churn-analysis.git](https://github.com/your-username/customer-churn-analysis.git)\n",
    "cd customer-churn-analysis\n",
    "\n",
    "# Create virtual environment & install requirements\n",
    "python -m venv venv\n",
    "source venv/bin/activate  # On Windows: venv\\Scripts\\activate\n",
    "pip install -r requirements.txt\n",
    "\n",
    "# Run full pipeline via main.py\n",
    "python main.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "216eaddc-1ee5-4138-bc4f-45e35244f969",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
