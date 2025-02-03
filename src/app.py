import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("models/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define user input function
def get_user_input():
    st.sidebar.header("Customer Information")
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 24)
    MonthlyCharges = st.sidebar.number_input("Monthly Charges", min_value=10, max_value=150, value=50)
    TotalCharges = st.sidebar.number_input("Total Charges", min_value=10, max_value=8000, value=1000)
    Contract = st.sidebar.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Years"])
    InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber Optic", "No"])
    PaymentMethod = st.sidebar.selectbox("Payment Method", ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"])
    PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

    user_data = pd.DataFrame(
        {
            "tenure": [tenure],
            "MonthlyCharges": [MonthlyCharges],
            "TotalCharges": [TotalCharges],
            "Contract": [Contract],
            "InternetService": [InternetService],
            "PaymentMethod": [PaymentMethod],
            "PaperlessBilling": [PaperlessBilling]
        }
    )
    return user_data

# Load user input
data = get_user_input()

# Encode categorical variables
encoder_dict = {
    "Contract": {"Month-to-Month": 0, "One Year": 1, "Two Years": 2},
    "InternetService": {"DSL": 0, "Fiber Optic": 1, "No": 2},
    "PaymentMethod": {"Electronic Check": 0, "Mailed Check": 1, "Bank Transfer": 2, "Credit Card": 3},
    "PaperlessBilling": {"Yes": 1, "No": 0}
}
for col, mapping in encoder_dict.items():
    data[col] = data[col].map(mapping)

# Make prediction
prediction = model.predict(data)[0]
prediction_prob = model.predict_proba(data)[0][1]

# Display prediction result
st.write("## Churn Prediction Result")
if prediction == 1:
    st.error(f"This customer is likely to churn (Probability: {prediction_prob:.2f})")
else:
    st.success(f"This customer is not likely to churn (Probability: {prediction_prob:.2f})")
