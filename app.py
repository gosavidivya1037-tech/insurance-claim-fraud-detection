import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessor
model = joblib.load("xgb_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Page settings
st.set_page_config(
    page_title="Insurance Claim Fraud Detection",
    page_icon="car"
)

# Title
st.title("Insurance Claim Fraud Detection")
st.write("Vehicle Insurance Claim Fraud Detection using XGBoost")

st.header("Claim Details")

# Month
month = st.selectbox(
    "Month",
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
)

# Week of Month
week_of_month = st.selectbox(
    "Week of Month",
    [1, 2, 3, 4, 5]
)

# Day of Week
day_of_week = st.selectbox(
    "Day of Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday",
     "Friday", "Saturday", "Sunday"]
)

# Vehicle Make
make = st.selectbox(
    "Vehicle Make",
    [
        "Accura", "BMW", "Chevrolet", "Dodge", "Ferrari",
        "Ford", "Honda", "Jaguar", "Lexus", "Mazda",
        "Mecedes", "Mercury", "Nissan", "Pontiac",
        "Porsche", "Saab", "Saturn", "Toyota", "Volkswagen"
    ]
)

# Accident Area
accident_area = st.selectbox(
    "Accident Area",
    ["Urban", "Rural"]
)

# Day of Week Claimed
day_of_week_claimed = st.selectbox(
    "Day of Week Claimed",
    ["Monday", "Tuesday", "Wednesday", "Thursday",
     "Friday", "Saturday", "Sunday", "0"]
)

# Month Claimed
month_claimed = st.selectbox(
    "Month Claimed",
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "0"]
)

# Week of Month Claimed
week_of_month_claimed = st.selectbox(
    "Week of Month Claimed",
    [1, 2, 3, 4, 5]
)
# -----------------------------
# Remaining Claim Details
# -----------------------------

# Sex
sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

# Marital Status
marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Widow", "Divorced"]
)

# Age
age = st.number_input(
    "Age",
    min_value=18,
    max_value=80,
    value=39
)

# Fault
fault = st.selectbox(
    "Fault",
    ["Policy Holder", "Third Party"]
)

# Policy Type
policy_type = st.selectbox(
    "Policy Type",
    [
        "Sedan - Collision",
        "Sedan - Liability",
        "Sedan - All Perils",
        "Sport - Collision",
        "Sport - Liability",
        "Sport - All Perils",
        "Utility - Collision",
        "Utility - Liability",
        "Utility - All Perils"
    ]
)

# Vehicle Category
vehicle_category = st.selectbox(
    "Vehicle Category",
    ["Sport", "Utility", "Sedan"]
)

# Vehicle Price
vehicle_price = st.selectbox(
    "Vehicle Price",
    [
        "less than 20000",
        "20000 to 29000",
        "30000 to 39000",
        "40000 to 59000",
        "60000 to 69000",
        "more than 69000"
    ]
)

# Representative Number
rep_number = st.number_input(
    "Representative Number",
    min_value=1,
    max_value=16,
    value=8
)

# Deductible
deductible = st.selectbox(
    "Deductible",
    [300, 400, 500, 600, 700]
)

# Driver Rating
driver_rating = st.selectbox(
    "Driver Rating",
    [1, 2, 3, 4]
)

# Days Policy Accident
days_policy_accident = st.selectbox(
    "Days Policy Accident",
    ["8 to 15", "15 to 30", "30 to 60", "more than 60",
     "none", "1 to 7"]
)

# Days Policy Claim
days_policy_claim = st.selectbox(
    "Days Policy Claim",
    ["8 to 15", "15 to 30", "more than 30", "none"]
)

# Past Number of Claims
past_number_claims = st.selectbox(
    "Past Number of Claims",
    ["none", "1", "2 to 4", "more than 4"]
)

# Age of Vehicle
age_vehicle = st.selectbox(
    "Age of Vehicle",
    ["new", "2 years", "3 years", "4 years", "5 years",
     "6 years", "7 years", "more than 7"]
)

# Age of Policy Holder
age_policy_holder = st.selectbox(
    "Age of Policy Holder",
    ["16 to 17", "18 to 20", "21 to 25", "26 to 30",
     "31 to 35", "36 to 40", "41 to 50", "51 to 65",
     "over 65"]
)

# Police Report Filed
police_report = st.selectbox(
    "Police Report Filed",
    ["Yes", "No"]
)

# Witness Present
witness_present = st.selectbox(
    "Witness Present",
    ["Yes", "No"]
)

# Agent Type
agent_type = st.selectbox(
    "Agent Type",
    ["External", "Internal"]
)

# Number of Supplements
number_supplements = st.selectbox(
    "Number of Supplements",
    ["none", "1", "2 to 5", "more than 5"]
)

# Address Change Claim
address_change = st.selectbox(
    "Address Change Claim",
    ["no change", "under 6 months", "6 months to 1 year",
     "1 to 3 years", "4 to 8 years"]
)

# Number of Cars
number_cars = st.selectbox(
    "Number of Cars",
    ["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"]
)

# Year
year = st.selectbox(
    "Year",
    [1994, 1995, 1996]
)

# Base Policy
base_policy = st.selectbox(
    "Base Policy",
    ["Liability", "Collision", "All Perils"]
)
# -----------------------------
# Prediction
# -----------------------------

if st.button("🔍 Predict Claim"):

    input_data = pd.DataFrame([{
        "Month": month,
        "WeekOfMonth": week_of_month,
        "DayOfWeek": day_of_week,
        "Make": make,
        "AccidentArea": accident_area,
        "DayOfWeekClaimed": day_of_week_claimed,
        "MonthClaimed": month_claimed,
        "WeekOfMonthClaimed": week_of_month_claimed,
        "Sex": sex,
        "MaritalStatus": marital_status,
        "Age": age,
        "Fault": fault,
        "PolicyType": policy_type,
        "VehicleCategory": vehicle_category,
        "VehiclePrice": vehicle_price,
        "RepNumber": rep_number,
        "Deductible": deductible,
        "DriverRating": driver_rating,
        "Days_Policy_Accident": days_policy_accident,
        "Days_Policy_Claim": days_policy_claim,
        "PastNumberOfClaims": past_number_claims,
        "AgeOfVehicle": age_vehicle,
        "AgeOfPolicyHolder": age_policy_holder,
        "PoliceReportFiled": police_report,
        "WitnessPresent": witness_present,
        "AgentType": agent_type,
        "NumberOfSuppliments": number_supplements,
        "AddressChange_Claim": address_change,
        "NumberOfCars": number_cars,
        "Year": year,
        "BasePolicy": base_policy
    }])

    # Transform input
    input_encoded = preprocessor.transform(input_data)

    # Prediction
    prediction = model.predict(input_encoded)[0]

    # Probability
    probability = model.predict_proba(input_encoded)[0][1]

    if prediction == 1:
        st.error("FRAUDULENT CLAIM")
        st.write(f"Fraud Probability: {probability:.2%}")
    else:
        st.success("GENUINE CLAIM")
        st.write(f"Fraud Probability: {probability:.2%}")
        