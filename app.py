import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# ---------------- LOAD MODELS ----------------
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
iso_model = joblib.load("iso_model.pkl")
scaler = joblib.load("scaler.pkl")
type_encoder = joblib.load("type_encoder.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    background-color: #f4f7fb;
}

[data-testid="stAppViewContainer"] {
    background-color: #f4f7fb;
}

.block-container {
    padding-top: 4rem;
    padding-bottom: 2rem;
}
[data-testid="stHorizontalBlock"] {
    gap: 1rem;
}

/* HEADER */
.main-title {
    font-size: 48px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 35px;
}

/* RESULT CARDS */
.result-card {
    background: white;
    padding: 24px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.05);
    border: 1px solid #e5e7eb;
}

.result-title {
    color: #6b7280;
    font-size: 15px;
    margin-bottom: 10px;
}

.result-value {
    font-size: 34px;
    font-weight: bold;
    color: #111827;
}

/* ALERTS */
.alert-high {
    background-color: #fee2e2;
    color: #b91c1c;
}

.alert-medium {
    background-color: #fef3c7;
    color: #92400e;
}

.alert-low {
    background-color: #dcfce7;
    color: #166534;
}

.final-alert {
    padding: 18px;
    border-radius: 18px;
    text-align: center;
    font-size: 24px;
    font-weight: 700;
    margin-top: 25px;
}

/* BUTTON */
.stButton>button {
    width: 100%;
    background-color: #2563eb;
    color: white;
    border-radius: 12px;
    height: 3.2em;
    border: none;
    font-size: 16px;
    font-weight: 600;
}

.stButton>button:hover {
    background-color: #1d4ed8;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    "<div class='main-title'>💳 AI Fraud Detection System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Real-time intelligent fintech fraud analysis</div>",
    unsafe_allow_html=True
)

# ---------------- LAYOUT ----------------
left, right = st.columns([1, 1.2])

# ---------------- INPUT SECTION ----------------
with left:
    # Removed the empty html <div class='input-card'> wrapper that was causing the blank card bug
    st.subheader("Transaction Details")

    transaction_type = st.selectbox(
        "Transaction Type",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=5000.0
    )

    oldbalanceOrg = st.number_input(
        "Sender Old Balance",
        min_value=0.0,
        value=10000.0
    )

    newbalanceOrig = st.number_input(
        "Sender New Balance",
        min_value=0.0,
        value=5000.0
    )

    oldbalanceDest = st.number_input(
        "Receiver Old Balance",
        min_value=0.0,
        value=2000.0
    )

    newbalanceDest = st.number_input(
        "Receiver New Balance",
        min_value=0.0,
        value=7000.0
    )

    predict = st.button("Analyze Transaction")

# ---------------- PREDICTION / RIGHT COLUMN ----------------
with right:
    if predict:
        # Encode type
        type_encoded = type_encoder.transform([transaction_type])[0]

        # Feature engineering
        sender_error = (oldbalanceOrg - newbalanceOrig - amount)
        receiver_error = (newbalanceDest - oldbalanceDest - amount)
        is_large_transaction = int(amount > 200000)

        # Input array
        input_data = np.array([[
            1, type_encoded, amount, oldbalanceOrg, newbalanceOrig,
            oldbalanceDest, newbalanceDest, sender_error, receiver_error, is_large_transaction
        ]])

        # Scale
        input_scaled = scaler.transform(input_data)

        # Predictions
        rf_prob = float(rf_model.predict_proba(input_scaled)[0][1])
        xgb_prob = float(xgb_model.predict_proba(input_scaled)[0][1])
        iso_score = float(-iso_model.decision_function(input_scaled)[0])
        iso_score = max(0, min(1, iso_score))

        # Base ML Risk
        ml_risk = (0.45 * rf_prob + 0.45 * xgb_prob + 0.10 * iso_score) * 100

        # Rule-Based Boosting
        rule_boost = 0
        if amount > 200000: rule_boost += 20
        if transaction_type in ["TRANSFER", "CASH_OUT"]: rule_boost += 20
        if abs(sender_error) > 1000: rule_boost += 25
        if abs(receiver_error) > 1000: rule_boost += 15
        if oldbalanceOrg > 0 and newbalanceOrig < 0.1 * oldbalanceOrg: rule_boost += 20

        # Final Risk
        risk_score = min(100, ml_risk + rule_boost)

        # Debugging Output
        st.write("ML Risk:", round(ml_risk, 2))
        st.write("Rule Boost:", rule_boost)
        st.write("Final Risk:", round(risk_score, 2))

        # Risk Threshold Flagging
        if risk_score >= 70:
            label = "HIGH RISK"
            alert_class = "alert-high"
        elif risk_score >= 40:
            label = "MEDIUM RISK"
            alert_class = "alert-medium"
        else:
            label = "LOW RISK"
            alert_class = "alert-low"

        # Explainable AI Text Generation
        reasons = []
        if amount > 200000: reasons.append("Large transaction amount")
        if sender_error != 0: reasons.append("Sender balance inconsistency")
        if receiver_error != 0: reasons.append("Receiver balance inconsistency")
        if transaction_type in ["TRANSFER", "CASH_OUT"]: reasons.append("High-risk transaction type")
        if not reasons: reasons.append("Transaction appears normal")

        # ---------------- RESULT CARDS ----------------
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div class='result-card'>
                <div class='result-title'>RF Probability</div>
                <div class='result-value'>{rf_prob:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class='result-card'>
                <div class='result-title'>XGB Probability</div>
                <div class='result-value'>{xgb_prob:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class='result-card'>
                <div class='result-title'>Risk Score</div>
                <div class='result-value'>{risk_score:.1f}</div>
            </div>
            """, unsafe_allow_html=True)

        # Final Alert Display
        st.markdown(f"""
        <div class='final-alert {alert_class}'>
            {label}
        </div>
        """, unsafe_allow_html=True)

        # Flag Explanations
        st.markdown("### Why was this flagged?")
        for reason in reasons:
            st.markdown(f"- {reason}")

    else:
        # This now matches perfectly inline with your transaction details layout
        st.info("Enter transaction details and click Analyze Transaction.")