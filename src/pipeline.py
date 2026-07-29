import joblib
import pandas as pd
import numpy as np

model = joblib.load("models/random_forest_tuned.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_defaults = joblib.load("models/feature_defaults.pkl")

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

def predict_intrusion_from_form(user_inputs: dict):
    """
    Takes a dict of user-provided raw feature values (only the subset 
    the form collects) and fills in the rest using training-data averages.
    Returns prediction + confidence.
    """
    # Start with defaults, then overwrite with whatever the user actually provided
    row = feature_defaults.copy()

    # Handle categorical inputs (protocol_type, service, flag) separately -
    # they need to become one-hot columns, not raw text
    categorical_inputs = {}
    for key in ["protocol_type", "service", "flag"]:
        if key in user_inputs:
            categorical_inputs[key] = user_inputs.pop(key)

    # Overwrite numeric defaults with user-provided numeric values
    for key, value in user_inputs.items():
        if key in row.index:
            row[key] = value

    # Apply the same log1p transform the raw inputs need (since defaults 
    # were computed AFTER log1p in Day 4, but raw user input is NOT log1p'd yet)
    import numpy as np
    for col in ["duration", "src_bytes", "dst_bytes"]:
        if col in user_inputs:
            row[col] = np.log1p(user_inputs[col])

    # Set the correct one-hot column to 1, others in that group to 0
    for key, value in categorical_inputs.items():
        # Zero out all columns for this category first
        matching_cols = [c for c in row.index if c.startswith(f"{key}_")]
        for c in matching_cols:
            row[c] = 0
        # Set the chosen one to 1 (if it's not the dropped baseline category)
        target_col = f"{key}_{value}"
        if target_col in row.index:
            row[target_col] = 1

    df_row = pd.DataFrame([row])
    df_scaled = scaler.transform(df_row)

    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    return ("attack" if prediction == 1 else "normal"), probability
