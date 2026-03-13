# Medical Insurance Charges Prediction

Predict medical insurance charges based on patient information using Python machine learning models.

---

## Overview
This project predicts insurance costs using features like **age, sex, BMI, number of children, smoking status, and region**. It includes:

- Exploratory Data Analysis (EDA)
- Data preprocessing (scaling & encoding)
- Model training and evaluation
- Prediction for new patients

---

## Dataset
The dataset `Medical_insurance.csv` includes:

| Column    | Description                       |
|-----------|-----------------------------------|
| age       | Age of patient                     |
| sex       | Gender (male/female)               |
| bmi       | Body Mass Index                     |
| children  | Number of children                  |
| smoker    | Smoking status (yes/no)             |
| region    | Residential region                  |
| charges   | Insurance charges (target)          |

---

## Installation
1. Clone the repo:  
```bash
git clone <repository-url>
cd <repository-folder>
```
2. Install required libraries:  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Usage
Run the script:  
```bash
python insurance_prediction.py
```
- EDA plots are saved: `eda_charges_distribution.png`, `eda_categorical_vs_charges.png`, `eda_numerical_vs_charges.png`
- Model comparison plot: `model_comparison.png`
- Enter new patient details to predict charges.  

**Example Input:**  
```
Age: 35
Sex: male
BMI: 28.5
Children: 2
Smoker: no
Region: northeast
```

**Output:**  
```
Predicted Insurance Charge: $4,562.30
```

---

## Models Used
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

---

## Evaluation Metrics
- **R-squared (R²)** – Variance explained by model  
- **Mean Absolute Error (MAE)** – Average absolute error  
- **Root Mean Squared Error (RMSE)** – Square root of mean squared error  

---

## Project Structure
```
.
├── app/
│   └── app.py              # Streamlit web application
├── src/
│   └── train_model.py      # Data processing and model training script
├── data/
│   └── Medical_insurance.csv # Original dataset
├── models/
│   └── insurance_model.pkl # Saved best model
├── outputs/                # Generated EDA and comparison plots
├── requirements.txt        # Project dependencies
└── README.md
```

---

## Features
- EDA and visualization of numerical & categorical features
- Data preprocessing using pipelines
- Train and compare multiple regression models
- Predict charges for new patients
- Save and reuse trained models (optional)

---

## Future Improvements
- Hyperparameter tuning
- Input validation
- Feature engineering (BMI/age groups)
- Interactive GUI/web interface
- Model explainability using SHAP or feature importance

---

## License
MIT License
