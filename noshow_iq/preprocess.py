import pandas as pd


def load_and_clean(filepath="data/KaggleV2.csv"):
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower().replace("-", "_") for c in df.columns]
    for col in df.columns:
        if "show" in col.lower():
            df.rename(columns={col: "no_show"}, inplace=True)
            break
    df["no_show"] = df["no_show"].map({"No": 0, "Yes": 1})
    df["scheduledday"] = pd.to_datetime(df["scheduledday"])
    df["appointmentday"] = pd.to_datetime(df["appointmentday"])
    df["days_in_advance"] = (df["appointmentday"] - df["scheduledday"]).dt.days
    df = df[df["days_in_advance"] >= 0]
    df["appt_day_of_week"] = df["appointmentday"].dt.dayofweek
    df = df[(df["age"] >= 0) & (df["age"] <= 115)]
    df.drop(columns=["patientid", "appointmentid", "scheduledday", "appointmentday"], inplace=True)
    df["gender"] = df["gender"].map({"F": 0, "M": 1})
    if "neighbourhood" in df.columns:
        df.drop(columns=["neighbourhood"], inplace=True)
    return df


def get_features_and_target(df):
    X = df.drop(columns=["no_show"])
    y = df["no_show"]
    return X, y
