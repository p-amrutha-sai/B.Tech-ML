
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# Load trained model
model = joblib.load("model/churn_model.pkl")

# Application title
st.title("Customer Churn Prediction")
st.write(
    "Enter customer details to predict whether "
    "the customer is likely to leave the service."
)

# Customer details form
with st.form("customer_form"):

    st.subheader("Customer Information")

    gender = st.selectbox(
        "Gender", ["Male", "Female"]
    )

    senior = st.selectbox(
        "Senior Citizen", [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.selectbox(
        "Has Partner?", ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Has Dependents?", ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)", 0, 72, 12
    )

    st.subheader("Services")

    phone = st.selectbox(
        "Phone Service", ["Yes", "No"]
    )

    multiple = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    st.subheader("Billing Information")

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.selectbox(
        "Paperless Billing", ["Yes", "No"]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=150.0,
        value=70.0
    )

    total = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )

    submit = st.form_submit_button(
        "Predict Churn"
    )

# Make prediction
if submit:

    # Create customer record
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    # Predict churn
    prediction = model.predict(customer)[0]

    # Get probability of churn
    probability = model.predict_proba(customer)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is not likely to churn")

    st.write(
        f"Estimated churn probability: {probability * 100:.2f}%"
    )

    st.progress(float(probability))

    st.caption(
        "This is an educational prediction, "
        "not a guarantee of customer behaviour."
    )
