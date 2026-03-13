import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train_model import load_data, perform_eda, train_models

st.set_page_config(page_title="Medical Insurance Predictor", page_icon="🏥", layout="wide")
st.title("🏥 Medical Insurance Charges Predictor")

# Load data and train models using your original functions
@st.cache_data
def get_data():
    return load_data()

@st.cache_resource
def get_models(df):
    return train_models(df)

df = get_data()

with st.spinner("Training models…"):
    results_df, final_pipeline, best_model_name = get_models(df)

# Sidebar navigation
page = st.sidebar.radio("Navigate", ["📊 EDA", "🤖 Model Comparison", "💰 Predict Charges"])

# ── PAGE 1: EDA — calls your perform_eda() but renders plots in Streamlit ─
if page == "📊 EDA":
    st.header("Exploratory Data Analysis")
    st.dataframe(df.head())
    st.dataframe(df.describe())

    # charges distribution
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['charges'], kde=True, bins=30, ax=ax)
    ax.set_title('Distribution of Insurance Charges')
    ax.set_xlabel('Charges'); ax.set_ylabel('Frequency')
    st.pyplot(fig); plt.close()

    # categorical features vs charges
    categorical_features = ['sex', 'smoker', 'region', 'children']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Categorical Features vs. Insurance Charges', fontsize=16)
    for i, feature in enumerate(categorical_features):
        row, col = i // 2, i % 2
        sns.boxplot(x=feature, y='charges', data=df, ax=axes[row, col])
        axes[row, col].set_title(f'Charges by {feature.capitalize()}')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # numerical features vs charges
    numerical_features = ['age', 'bmi']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, feature in enumerate(numerical_features):
        sns.scatterplot(x=feature, y='charges', data=df, hue='smoker', alpha=0.7, ax=axes[i])
        axes[i].set_title(f'Charges vs. {feature.capitalize()}')
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ── PAGE 2: Model Comparison ──────────────────────────────────────────────
elif page == "🤖 Model Comparison":
    st.header("Model Comparison")
    st.dataframe(
        results_df.style
            .format({'R-squared': '{:.4f}', 'MAE': '${:,.2f}', 'RMSE': '${:,.2f}'})
            .highlight_max(subset=['R-squared'], color='lightgreen')
            .highlight_min(subset=['MAE', 'RMSE'], color='lightgreen')
    )
    st.success(f"✅ Best model: **{best_model_name}** (R² = {results_df.loc[best_model_name, 'R-squared']:.4f})")

    fig, ax = plt.subplots(figsize=(10, 6))
    results_df['R-squared'].plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title('Model Comparison by R-squared Score')
    ax.set_ylabel('R-squared Score')
    plt.xticks(rotation=45); plt.tight_layout()
    st.pyplot(fig); plt.close()

# ── PAGE 3: Predict — replaces input() with Streamlit widgets ────────────
elif page == "💰 Predict Charges":
    st.header("Predict Insurance Charges for a New Patient")
    st.info(f"Using best model: **{best_model_name}** (retrained on full dataset)")

    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            # Step 8.5: Get user input
            age       = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
            sex       = st.selectbox("Sex", ["male", "female"])
            height_m  = st.number_input("Height (meters)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
            weight_kg = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.5)
        with col2:
            children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
            smoker   = st.selectbox("Smoker?", ["no", "yes"])
            region   = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
        submitted = st.form_submit_button("💰 Predict Charge")

    if submitted:
        # Step 8.5.1: Calculate BMI (same as your original code)
        bmi = weight_kg / (height_m ** 2)

        # Step 8.5.2: Create a DataFrame with all the collected data (same as your original code)
        new_patient_data = pd.DataFrame({
            'age': [age], 'sex': [sex], 'bmi': [bmi],
            'children': [children], 'smoker': [smoker], 'region': [region]
        })

        # Step 8.6 / Step 9: Predict and display (same as your original code)
        predicted_charge = final_pipeline.predict(new_patient_data)[0]
        USD_TO_INR = 83.5
        predicted_charge_inr = predicted_charge * USD_TO_INR

        st.success(f"💰 Predicted Insurance Charge: **${predicted_charge:,.2f}** &nbsp;|&nbsp; **₹{predicted_charge_inr:,.2f}**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Charge (USD)", f"${predicted_charge:,.2f}")
        col2.metric("Charge (INR)", f"₹{predicted_charge_inr:,.2f}")
        col3.metric("Your BMI", f"{bmi:.2f}")
        st.dataframe(new_patient_data)