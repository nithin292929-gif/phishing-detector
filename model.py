import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import os

def train_model():
    df = pd.read_csv("emails.csv")
    df.columns = df.columns.str.strip()

    label_col = "Prediction"
    drop_cols = ["Email No.", label_col]

    X = df.drop(columns=drop_cols)
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    feature_cols = list(X.columns)
    with open("features.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    return {
        "accuracy": round(acc * 100, 2),
        "confusion_matrix": cm.tolist(),
        "labels": ["Safe", "Spam"]
    }

def predict_email(text):
    if not os.path.exists("model.pkl"):
        return {"error": "Model not trained yet"}

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("features.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    words = text.lower().split()
    row = {col: words.count(col) for col in feature_cols}
    df = pd.DataFrame([row])

    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    confidence = round(max(proba) * 100, 2)

    return {
        "prediction": "1" if prediction == 1 else "0",
        "confidence": confidence
    }