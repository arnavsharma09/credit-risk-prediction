import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib

# 1. Page config MUST be the absolute first Streamlit command
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="🏦",
    layout="wide"
)

# Force matplotlib to use a non-interactive backend suited for cloud servers
matplotlib.use('Agg')

# 2. Dynamic, environment-agnostic path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'rf_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')

# Safely load binary components
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    st.error(f"🚨 Model assets missing from deployment directory! Looking in: {BASE_DIR}")
    st.stop()

# Feature alignment signature matching model parameters
feature_columns = [
    'credit_utilization', 'age', 'late_30_59', 'debt_ratio',
    'monthly_income', 'open_loans', 'late_90', 'real_estate_loans',
    'late_60_89', 'dependents', 'debt_to_income',
    'total_late_payments', 'income_per_dependent', 'utilization_bucket'
]

# UI Title Block
st.title("🏦 Credit Risk Prediction System")
st.markdown("Enter a customer's financial details to predict their loan default probability.")
st.markdown("---")

# Layout Matrix Configuration
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Info")
    age = st.slider("Age", min_value=18, max_value=100, value=35)
    dependents = st.slider("Number of Dependents", min_value=0, max_value=10, value=0)
    monthly_income = st.number_input("Monthly Income ($)", min_value=0, max_value=100000, value=5000, step=500)

with col2:
    st.subheader("Credit Info")
    credit_utilization = st.slider("Credit Utilization (0 = none, 1 = maxed out)", 
                                    min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    debt_ratio = st.slider("Debt Ratio", min_value=0.0, max_value=10.0, value=0.3, step=0.01)
    open_loans = st.slider("Number of Open Loans", min_value=0, max_value=30, value=5)
    real_estate_loans = st.slider("Number of Real Estate Loans", min_value=0, max_value=10, value=1)

with col3:
    st.subheader("Payment History")
    late_30_59 = st.slider("Times 30-59 Days Late", min_value=0, max_value=15, value=0)
    late_60_89 = st.slider("Times 60-89 Days Late", min_value=0, max_value=15, value=0)
    late_90 = st.slider("Times 90+ Days Late", min_value=0, max_value=15, value=0)

st.markdown("---")

# Execution trigger
if st.button("🔍 Predict Default Risk", use_container_width=True):

    # Operational Feature Engineering
    def get_utilization_bucket(val):
        if val < 0.3:
            return 0
        elif val < 0.7:
            return 1
        else:
            return 2

    debt_to_income = debt_ratio * monthly_income
    total_late_payments = late_30_59 + late_60_89 + late_90
    income_per_dependent = monthly_income / (dependents + 1)
    util_bucket = get_utilization_bucket(credit_utilization)

    # Structuring payload input array
    input_data = pd.DataFrame([{
        'credit_utilization': credit_utilization,
        'age': age,
        'late_30_59': late_30_59,
        'debt_ratio': debt_ratio,
        'monthly_income': monthly_income,
        'open_loans': open_loans,
        'late_90': late_90,
        'real_estate_loans': real_estate_loans,
        'late_60_89': late_60_89,
        'dependents': dependents,
        'debt_to_income': debt_to_income,
        'total_late_payments': total_late_payments,
        'income_per_dependent': income_per_dependent,
        'utilization_bucket': util_bucket
    }])

    # Preprocessing scaling pass
    input_scaled = scaler.transform(input_data)

    # Model inference evaluation
    default_prob = model.predict_proba(input_scaled)[0][1]
    default_pct = default_prob * 100

    st.markdown("## Prediction Result")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if default_pct < 20:
            st.success("### ✅ Low Risk")
            st.markdown(f"**Default Probability: {default_pct:.1f}%**")
            st.markdown("This customer is unlikely to default on their loan.")
        elif default_pct < 50:
            st.warning("### ⚠️ Medium Risk")
            st.markdown(f"**Default Probability: {default_pct:.1f}%**")
            st.markdown("This customer has a moderate chance of defaulting.")
        else:
            st.error("### ❌ High Risk")
            st.markdown(f"**Default Probability: {default_pct:.1f}%**")
            st.markdown("This customer has a high chance of defaulting.")

    with res_col2:
        # horizontal gauge chart construction
        fig, ax = plt.subplots(figsize=(5, 1.5))
        ax.barh(['Risk'], [default_pct], color=(
            '#2ecc71' if default_pct < 20 else
            '#f39c12' if default_pct < 50 else
            '#e74c3c'
        ), height=0.4)
        ax.barh(['Risk'], [100 - default_pct], left=default_pct, color='#ecf0f1', height=0.4)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Default Probability (%)')
        ax.axvline(x=20, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(x=50, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title(f'Risk Gauge: {default_pct:.1f}%')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Explainable AI interpretation layer
    st.markdown("---")
    st.markdown("## Why did the model make this prediction?")
    st.markdown("The chart below shows which factors increased or decreased the default risk for this customer.")

    explainer = shap.TreeExplainer(model)
    input_df = pd.DataFrame(input_scaled, columns=feature_columns)
    shap_values = explainer.shap_values(input_df)
    
    # Versioning format normalization for matrix shape mutations
    if isinstance(shap_values, list):
        shap_values_default = shap_values[1]
    else:
        shap_values_default = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values

    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

    shap_explanation = shap.Explanation(
        values=shap_values_default[0],
        base_values=base_val,
        data=input_df.iloc[0],
        feature_names=feature_columns
    )

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(shap_explanation, show=False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # Data Presentation Grid
    st.markdown("---")
    st.markdown("## Customer Summary")
    summary_data = {
        'Feature': ['Age', 'Monthly Income', 'Credit Utilization', 'Total Late Payments',
                    'Debt Ratio', 'Dependents', 'Open Loans'],
        'Value': [age, f"${monthly_income:,}", f"{credit_utilization:.0%}",
                  total_late_payments, debt_ratio, dependents, open_loans]
    }
    st.table(pd.DataFrame(summary_data))