import streamlit as st
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
page = st.sidebar.radio("Go to", ["Home", "Predict", "Dashboard"])

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
    st.caption("Fields not shown below are automatically filled using average values from the training dataset.")

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
    # --- Basic input validation ---
    validation_errors = []

    if flag == "SF" and src_bytes == 0 and dst_bytes == 0 and duration == 0:
        validation_errors.append(
            "A 'SF' (successful) connection with zero duration and zero bytes is unusual - "
            "results may not be meaningful."
        )

    if src_bytes > 100_000_000 or dst_bytes > 100_000_000:
        validation_errors.append(
            "Byte values this large are extreme outliers rarely seen in real traffic - "
            "prediction confidence may be unreliable."
        )

    for warning in validation_errors:
        st.warning(f"⚠️ {warning}")
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

        try:
            with st.spinner("Analyzing traffic..."):
                prediction, confidence = predict_intrusion_from_form(user_inputs)

            st.subheader("Prediction Result")

            if prediction == "attack":
                st.error("⚠️ Attack Detected")
            else:
                st.success("✅ Normal Traffic")

            st.metric(label="Confidence", value=f"{confidence:.2%}")

        except Exception as e:
            st.error(f"Something went wrong while making the prediction: {e}")
            st.info("Try adjusting your inputs and predicting again.")

elif page == "Dashboard":
    st.title("📊 Model Dashboard")

    # --- Model Comparison Table ---
    st.subheader("Model Comparison")
    comparison_data = {
        "Model": ["Decision Tree", "Random Forest", "Logistic Regression", "Random Forest (Tuned)"],
        "Accuracy": [0.9986, 0.9990, 0.9735, 0.9990],
        "Precision": [0.9982, 0.9994, 0.9778, 0.9995],
        "Recall": [0.9987, 0.9985, 0.9650, 0.9985],
        "F1 Score": [0.9985, 0.9989, 0.9713, 0.9990],
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

    # --- Attack Type Distribution ---
    st.subheader("Attack Type Distribution (Training Data)")

    columns = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
        "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
        "root_shell","su_attempted","num_root","num_file_creations","num_shells",
        "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
        "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
        "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
        "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
        "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
        "label","difficulty"
    ]
    df_raw = pd.read_csv("data/KDDTrain+.txt", names=columns)

    fig, ax = plt.subplots(figsize=(10, 5))
    df_raw["label"].value_counts().plot(kind="bar", ax=ax)
    ax.set_yscale("log")
    ax.set_title("Attack Type Distribution (log scale)")
    st.pyplot(fig)

    # --- Binary Class Balance ---
    st.subheader("Normal vs Attack Balance")
    df_raw["binary_label"] = df_raw["label"].apply(lambda x: "normal" if x == "normal" else "attack")

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df_raw, x="binary_label", ax=ax2)
    ax2.set_title("Normal vs Attack Count")
    st.pyplot(fig2)

    # --- Feature Importance ---
    st.subheader("Top 15 Most Important Features")

    import joblib
    model = joblib.load("models/random_forest_tuned.pkl")
    feature_names = joblib.load("models/feature_defaults.pkl").index

    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_features = importances.sort_values(ascending=False).head(15)

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    top_features.sort_values().plot(kind="barh", ax=ax3)
    ax3.set_title("Top 15 Feature Importances (Random Forest, Tuned)")
    ax3.set_xlabel("Importance")
    st.pyplot(fig3)

    