import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

from noshow_iq.preprocess import load_and_clean, get_features_and_target

MODEL_PATH = "model.joblib"


def train(filepath="data/KaggleV2.csv"):
    df = load_and_clean(filepath)
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, report


def predict(features: dict):
    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([features])
    proba = model.predict_proba(df)[0][1]
    risk = "high" if proba >= 0.5 else "low"
    recommendation = (
        "Send reminder and call patient"
        if risk == "high"
        else "Standard reminder is enough"
    )
    return {"risk_level": risk, "probability": round(proba, 3),
            "recommendation": recommendation}


def evaluate(filepath="data/KaggleV2.csv"):
    from noshow_iq.preprocess import load_and_clean, get_features_and_target
    df = load_and_clean(filepath)
    X, y = get_features_and_target(df)
    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X)
    return classification_report(y, y_pred, output_dict=True)
