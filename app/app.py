import streamlit as st
import sys
import pandas as pd

sys.path.append("src")

from pipeline import predict_intrusion_from_form

st.set_page_config(
    page_title="AI Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)

# =========================
# Sidebar
# =========================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict"])

# =========================
# Home Page
# =========================
if page == "Home":

    st.title("🛡️ AI-Powered Network Intrusion Detection")

    st.markdown("""
This app uses a Machine Learning model trained on the **NSL-KDD Dataset**
to classify network traffic as **Normal** or **Attack**.

### How it works

- Random Forest Classifier
- Trained on NSL-KDD dataset
- Tested on official holdout dataset
- Predicts whether network traffic is normal or malicious

Use the sidebar to go to the **Predict** page.
""")

# =========================
# Predict Page
# =========================
elif page == "Predict":

    st.title("🔍 Predict Network Traffic")

    st.write("Enter the connection details below.")

    col1, col2 = st.columns(2)

    with col1:

        duration = st.number_input(
            "Duration",
            min_value=0,
            value=0
        )

        protocol_type = st.selectbox(
            "Protocol Type",
            ["tcp", "udp", "icmp"]
        )

        service = st.selectbox(
            "Service",
            ["http", "ftp_data", "smtp", "telnet", "private", "other"]
        )

        flag = st.selectbox(
            "Flag",
            ["SF", "S0", "REJ", "RSTR", "RSTO"]
        )

        src_bytes = st.number_input(
            "Source Bytes",
            min_value=0,
            value=0
        )

        dst_bytes = st.number_input(
            "Destination Bytes",
            min_value=0,
            value=0
        )

    with col2:

        count = st.number_input(
            "Count",
            min_value=0,
            value=0
        )

        srv_count = st.number_input(
            "Service Count",
            min_value=0,
            value=0
        )

        logged_in = st.selectbox(
            "Logged In",
            [0, 1]
        )

        serror_rate = st.slider(
            "SYN Error Rate",
            0.0,
            1.0,
            0.0
        )

        same_srv_rate = st.slider(
            "Same Service Rate",
            0.0,
            1.0,
            0.0
        )

    # =========================
    # Predict Button
    # =========================

    if st.button("Predict"):

        user_inputs = {
            "duration": duration,
            "protocol_type": protocol_type,
            "service": service,
            "flag": flag,
            "src_bytes": src_bytes,
            "dst_bytes": dst_bytes,
            "count": count,
            "srv_count": srv_count,
            "logged_in": logged_in,
            "serror_rate": serror_rate,
            "same_srv_rate": same_srv_rate
        }

        prediction, confidence = predict_intrusion_from_form(user_inputs)

        st.subheader("Prediction Result")

        if prediction == "attack":
            st.error("⚠️ Attack Detected")
        else:
            st.success("✅ Normal Traffic")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2%}"
        )