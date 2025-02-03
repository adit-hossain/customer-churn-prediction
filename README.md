# Customer Churn Prediction Project

## Overview
This project predicts customer churn for a telecom company using machine learning. It includes data preprocessing, exploratory data analysis (EDA), feature engineering, and training multiple models to determine the best performing model. The best model is deployed using a Streamlit web application, and the dataset is prepared for visualization in Power BI/Tableau.

## Features
- Data Preprocessing (Handling missing values, encoding categorical variables)
- Exploratory Data Analysis (EDA) with visualizations
- Feature Engineering & Selection
- Handling class imbalance using SMOTE
- Training ML Models: Random Forest, Logistic Regression, and XGBoost
- Model Evaluation (Confusion Matrix, ROC-AUC Score)
- Model Deployment using Streamlit
- Power BI/Tableau Dashboard for Business Insights

## Dataset
The dataset consists of customer information with features like:
- **Demographics** (`gender`, `SeniorCitizen`, `Partner`, `Dependents`)
- **Service Subscriptions** (`InternetService`, `PhoneService`, `StreamingTV`)
- **Account Details** (`Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`)
- **Target Variable:** `Churn` (Yes/No)

## Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction
   ```
2. **Create a Virtual Environment & Install Dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Running the Jupyter Notebook
To train the model and generate insights:
```bash
jupyter notebook
```
Open `churn_prediction_notebook.ipynb` and run all cells.

## Running the Streamlit App
To deploy the customer churn prediction model:
```bash
streamlit run src/app.py
```

## Power BI/Tableau Dashboard
- The dataset `cleaned_customer_churn.csv` is available for visualization.
- Upload the dataset to Power BI/Tableau and create churn-related dashboards.

## Results
The best model is selected based on accuracy and ROC-AUC score. The model predicts whether a customer is likely to churn based on input features.

## Future Improvements
- Hyperparameter tuning for better performance.
- Adding more customer behavioral data for improved predictions.
- Deploying as a web API for real-time churn prediction.

## Author
**Your Name**  
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)  
- GitHub: [Your Repository](https://github.com/yourusername)
