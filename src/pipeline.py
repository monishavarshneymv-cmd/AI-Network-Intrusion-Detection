import joblib
import pandas as pd
import numpy as np

model = joblib.load("models/random_forest_tuned.pkl")
scaler = joblib.load("models/scaler.pkl")

df_train_full = pd.read_csv("data/processed_train.csv")
train_columns = df_train_full.drop(columns=["binary_label"]).columns

def predict_intrusion(raw_row_df):
    """
    Takes a raw dataframe with the original 41 feature columns
    (no label/difficulty) and returns predictions + confidence scores.
    """
    df = raw_row_df.copy()

    for col in ["duration", "src_bytes", "dst_bytes"]:
        df[col] = np.log1p(df[col])

    df = pd.get_dummies(df, columns=["protocol_type", "service", "flag"], drop_first=True)
    df = df.reindex(columns=train_columns, fill_value=0)
    df_scaled = scaler.transform(df)

    predictions = model.predict(df_scaled)
    probabilities = model.predict_proba(df_scaled)[:, 1]

    return pd.DataFrame({
        "prediction": ["attack" if p == 1 else "normal" for p in predictions],
        "confidence": probabilities
    })