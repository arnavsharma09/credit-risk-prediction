# Credit Risk Prediction System

A machine learning project that predicts whether a loan applicant is likely to default using real-world financial data. Features a live interactive web dashboard built with Streamlit that uses SHAP to explain the model's decisions.

Live App: https://arnavsharma-credit-risk.streamlit.app/

---

## The Goal
Banks need to know the risk of lending money before approving a loan. This project uses historical customer data to predict if an applicant will experience serious financial delinquency (default) within 2 years.

---

## The Data
- Source: Kaggle (Give Me Some Credit dataset)
- Size: 150,000 rows, 11 features
- Target: SeriousDlqin2yrs (1 = defaulted, 0 = paid on time)
- Challenge: Extreme class imbalance. Only 7% of the rows represent actual defaults, meaning the model had to be trained carefully to avoid ignoring the minority class.

---

## Project Workflow
1. Data Cleaning: Fixed missing values using median imputation and cleared out impossible data points.
2. Feature Engineering: Created practical risk metrics like debt_to_income, total_late_payments, and income_per_dependent.
3. Handling Imbalance: Used SMOTE to generate synthetic data for the minority default class so the model could learn patterns effectively.
4. Model Evaluation: Compared Logistic Regression, XGBoost, and Random Forest using AUC-ROC scores to find the strongest predictor.
5. Explainability: Integrated SHAP to break down the model's inner logic into visual waterfall charts for the end user.

---

## Model Benchmarks

| Model | AUC-ROC Score |
| :--- | :---: |
| Logistic Regression | 0.8234 |
| XGBoost | 0.8492 |
| **Random Forest (Tuned)** | **0.8588** |

The Random Forest model performed the best with an AUC-ROC of 0.8588 and was exported to power the production web app.

---

## Key Takeaways
- Credit Utilization: Maxing out credit limits relative to total allowances is the single biggest predictor of default risk.
- Late Payments: Missing payments for 30-59 days strongly spikes the risk score, but crossing the 90-day mark causes it to skyrocket.
- Age Trends: In this dataset, younger applicants (under 35) statistically showed a higher tendency to default compared to older age brackets.

---

## Tech Stack
- Data & Plots: Pandas, NumPy, Seaborn, Matplotlib
- Machine Learning: Scikit-Learn, XGBoost, Imbalanced-Learn
- Deployment: Streamlit, Joblib, SHAP

---

## Folder Structure
```text
credit-risk-prediction/
├── app.py                     # Live Streamlit application code
├── credit_risk.ipynb          # Training notebook (EDA, SMOTE, and Modeling)
├── cs-training.csv            # Training data file
├── requirements.txt           # App dependencies
├── rf_model.pkl               # Saved Random Forest model
├── scaler.pkl                 # Saved data scaler
└── README.md                  # Project documentation
