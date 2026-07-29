import streamlit as st

st.set_page_config(
    page_title="AI Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict"])

# --- Home Page ---
if page == "Home":
    st.title("🛡️ AI-Powered Network Intrusion Detection")
    st.markdown("""
    This app uses a machine learning model trained on the **NSL-KDD** dataset 
    to classify network connections as **normal** or **attack**.

    ### How it works
    - Trained a Random Forest classifier on network traffic features
    - Achieved ~99.9% accuracy on internal test data
    - Validated on NSL-KDD's official holdout test set for honest generalization testing

    Use the sidebar to navigate to the **Predict** page and try it yourself.
    """)

# --- Prediction Page (skeleton only for now) ---
elif page == "Predict":
    st.title("🔍 Predict Network Traffic")
    st.write("Prediction functionality will be added here in Day 11.")