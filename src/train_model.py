#Step 1: Import Libraries
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'Medical_insurance.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    #Step 2: Loading the dataset
    df = pd.read_csv(DATA_PATH)
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Information:")
    df.info()
    print("\nChecking for missing values:")
    print(df.isnull().sum())
    print("\nStatistical Summary:")
    print(df.describe())
    return df


def perform_eda(df):
    #Step 3: Performing Exploratory Data Analysis (EDA)
    print("--- Performing EDA ---")
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))
    sns.histplot(df['charges'], kde=True, bins=30)
    plt.title('Distribution of Insurance Charges')
    plt.xlabel('Charges')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(OUTPUT_DIR, 'eda_charges_distribution.png'))
    plt.close()

    #Step 3.1: Find the relationship between categorical features and charges
    categorical_features = ['sex', 'smoker', 'region', 'children']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Categorical Features vs. Insurance Charges', fontsize=16)
    for i, feature in enumerate(categorical_features):
        row, col = i // 2, i % 2
        sns.boxplot(x=feature, y='charges', data=df, ax=axes[row, col])
        axes[row, col].set_title(f'Charges by {feature.capitalize()}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'eda_categorical_vs_charges.png'))
    plt.close()

    #Step 3.2: Relationship between numerical features and charges
    numerical_features = ['age', 'bmi']
    plt.figure(figsize=(14, 6))
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(1, 2, i)
        sns.scatterplot(x=feature, y='charges', data=df, hue='smoker', alpha=0.7)
        plt.title(f'Charges vs. {feature.capitalize()}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'eda_numerical_vs_charges.png'))
    plt.close()


def train_models(df):
    #Step 4: Data Preprocessing
    print("--- Preprocessing Data ---")
    X = df.drop('charges', axis=1)
    y = df['charges']

    #Step 4.1: Identify categorical and numerical features
    categorical_cols = ['sex', 'smoker', 'region', 'children']
    numerical_cols = ['age', 'bmi']

    #Step 4.2: Create a preprocessor using ColumnTransformer (OneHotEncoder for categorical data, StandardScaler for numerical data)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    #Step 4.3: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    #Step 5: Model Training & Evaluation
    results = {}
    models = {}

    #Step 5.1.1: Model 1: Linear Regression
    name = "Linear Regression"
    model = LinearRegression()
    models[name] = model
    print(f"--- Training: {name} ---")
    lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    lr_pipeline.fit(X_train, y_train)
    y_pred = lr_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {'R-squared': r2, 'MAE': mae, 'RMSE': rmse}
    print(f"R-squared: {r2:.4f} \nMAE: ${mae:,.2f} \nRMSE: ${rmse:,.2f}")

    #Step 5.1.2: Model 2: Decision Tree
    name = "Decision Tree"
    model = DecisionTreeRegressor(random_state=42)
    models[name] = model
    print(f"--- Training: {name} ---")
    dt_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    dt_pipeline.fit(X_train, y_train)
    y_pred = dt_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {'R-squared': r2, 'MAE': mae, 'RMSE': rmse}
    print(f"R-squared: {r2:.4f} \nMAE: ${mae:,.2f} \nRMSE: ${rmse:,.2f}")

    #Step 5.1.3: Model 3: Random Forest
    name = "Random Forest"
    model = RandomForestRegressor(random_state=42)
    models[name] = model
    print(f"--- Training: {name} ---")
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    rf_pipeline.fit(X_train, y_train)
    y_pred = rf_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {'R-squared': r2, 'MAE': mae, 'RMSE': rmse}
    print(f"R-squared: {r2:.4f} \nMAE: ${mae:,.2f} \nRMSE: ${rmse:,.2f}")

    #Step 5.1.4: Model 4: Gradient Boosting
    name = "Gradient Boosting"
    model = GradientBoostingRegressor(random_state=42)
    models[name] = model
    print(f"--- Training: {name} ---")
    gb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    gb_pipeline.fit(X_train, y_train)
    y_pred = gb_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {'R-squared': r2, 'MAE': mae, 'RMSE': rmse}
    print(f"R-squared: {r2:.4f} \nMAE: ${mae:,.2f} \nRMSE: ${rmse:,.2f}")

    #Step 6: Final Model Comparison
    print("--- Final Model Comparison ---")
    #Step 6.1: Convert results to a DataFrame for easy viewing
    results_df = pd.DataFrame(results).T.sort_values(by='R-squared', ascending=False)
    print(results_df)

    #Step 6.2: Visualize the results
    results_df['R-squared'].plot(kind='bar', figsize=(10, 6), color='skyblue')
    plt.title('Model Comparison by R-squared Score')
    plt.ylabel('R-squared Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'))
    plt.close()

    #Step 7: Predicting the best model
    best_model_name = results_df.index[0]
    best_model_score = results_df.loc[best_model_name]
    print(f"The best performing model is '{best_model_name}' with the following scores:")
    print(best_model_score)
    print("\nThis model is recommended for predicting medical insurance prices.")

    #Step 8: Predict the charges for a new patient
    #Step 8.1: Retrain the Best Model on the Full Dataset ---
    best_model = models[best_model_name]
    #Step 8.2: Create the final pipeline object
    final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', best_model)])
    #Step 8.3: Train it on ALL the data (X, y)
    final_pipeline.fit(X, y)
    print(f"Final model ('{best_model_name}') is trained and ready.")
    
    model_path = os.path.join(MODEL_DIR, "insurance_model.pkl")
    joblib.dump(final_pipeline, model_path)

    print(f"Model saved successfully to {model_path}!")

    return results_df, final_pipeline, best_model_name


def predict_charges(final_pipeline):
    #Step 8.4: Enter New Patient Details ---
    print("--- Enter New Patient Details ---")
    #Step 8.5: Get user input and create a DataFrame
    age = int(input("Enter age: "))
    sex = input("Enter sex (male/female): ").lower()
    height_m = float(input("Enter height in meters (e.g., 1.75): "))
    weight_kg = float(input("Enter weight in kilograms (e.g., 70): "))
    #Step 8.5.1: Calculate and print BMI immediately
    bmi = weight_kg / (height_m ** 2)
    print(f"Your calculated BMI is: {bmi:.2f}")
    children = int(input("Enter number of children: "))
    smoker = input("Are you a smoker? (yes/no): ").lower()
    region = input("Enter region (e.g., southwest): ").lower()
    #Step 8.5.2: Create a DataFrame with all the collected data
    new_patient_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    #Step 8.6: Predict the charge using the final model
    predicted_charge = final_pipeline.predict(new_patient_data)[0]
    #Step 9: Display the final result (USD + INR)
    USD_TO_INR = 83.5
    predicted_charge_inr = predicted_charge * USD_TO_INR
    print(f"Predicted Insurance Charge: ${predicted_charge:,.2f} (₹{predicted_charge_inr:,.2f})")
    return predicted_charge


# ── Run as script (original behaviour preserved) ──────────────────────────
if __name__ == "__main__":
    df = load_data()
    perform_eda(df)
    results_df, final_pipeline, best_model_name = train_models(df)
    predict_charges(final_pipeline)