import streamlit as st
import pandas as pd
import pickle

# ---------------------------
# 1. PAGE HEADER
# ---------------------------
st.title("Credit Approval Prediction App")
st.write("This app predicts whether a credit card application is likely to be approved or rejected.")

# ---------------------------
# 2. USER INPUT FUNCTION
# ---------------------------
def user_input():
    col_1, col_2 = st.columns(2)

    debt = col_1.slider('How much Debt?', min_value=0.0, max_value=100.0, value=0.0)
    yearsEmployed = col_1.slider('Years of employment', min_value=0.0, max_value=100.0, value=0.0)
    income = col_1.slider('Total income per year', min_value=10.0, max_value=100000.0, value=10.0)

    priorDefault = col_2.selectbox('Prior default? (1 = True, 0 = False)', options=[0,1], index=0)
    employed = col_2.selectbox('Employment status (0 = Employed, 1 = Not employed)', options=[0,1], index=1)
    creditScore = col_2.slider("Customer's credit score", min_value=0.0, max_value=100.0, value=0.0)

    # MUST MATCH EXACT TRAINING FEATURE NAMES
    input_dict = {
        'Debt': float(debt),
        'YearsEmployed': float(yearsEmployed),
        'PriorDefault': int(priorDefault),
        'Employed': int(employed),
        'Income': float(income),
        'CreditScore': float(creditScore)
    }

    df = pd.DataFrame([input_dict])
    return df

# Collect user input
df = user_input()

# Show input
st.subheader("User Input Parameters")
st.write(df)

# ---------------------------
# 3. LOAD MODEL, SCALER, FEATURES
# ---------------------------
logreg = pickle.load(open("creditApproval_model.sav", "rb"))
scaler = pickle.load(open("scaler.sav", "rb"))

import pickle as pkl
FEATURES = pkl.load(open("features_list.pkl", "rb"))   # ['Debt','YearsEmployed',...]

# ---------------------------
# 4. PREPARE & SCALE INPUT
# ---------------------------
# Verify required features exist
missing = [c for c in FEATURES if c not in df.columns]
if missing:
    st.error(f"Error: Missing required input fields: {missing}")
else:
    df_input = df[FEATURES].copy()

    # Scale using the fitted MinMaxScaler
    X_scaled = scaler.transform(df_input)

    # ---------------------------
    # 5. PREDICT
    # ---------------------------
    pred = logreg.predict(X_scaled)[0]

    # Probability (optional)
    if hasattr(logreg, "predict_proba"):
        proba = logreg.predict_proba(X_scaled)[0,1]
    else:
        proba = None

    # ---------------------------
    # 6. OUTPUT RESULTS
    # ---------------------------
    st.subheader("Prediction")
    if pred == 1:
        st.success("Your credit request is **Approved**")
    else:
        st.error("Your credit request is **Rejected**")

    if proba is not None:
        st.write(f"Approval Probability: **{proba:.2f}**")
