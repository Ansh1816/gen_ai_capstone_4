"""
model_brain.py
--------------
Exact replication of genaicapstone.py pipeline.
Trains DecisionTreeClassifier(max_depth=10) and saves:
  - model.pkl   (trained Decision Tree)
  - scaler.pkl  (fitted StandardScaler)
  - feature_cols.pkl (ordered list of feature columns used in training)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
import os

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "KaggleV2-May-2016.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_cols.pkl")


def load_and_preprocess(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Exact preprocessing from genaicapstone.py:
      1. Encode No-show (No→0, Yes→1)
      2. Handicap dummies
      3. Age filter [0, 100]
      4. AwaitingTime (abs days between ScheduledDay and AppointmentDay)
      5. Num_App_Missed (cumulative prior no-shows per patient)
      6. Drop unused columns
      7. Build feature matrix X with exactly these columns:
         Gender, Diabetes, Hipertension, Scholarship, SMS_received,
         Handicap_0..4, Num_App_Missed, Age, AwaitingTime
    """
    df = pd.read_csv(csv_path)

    # Encode target
    df["No-show"].replace("No", 0, inplace=True)
    df["No-show"].replace("Yes", 1, inplace=True)

    # Handicap dummies
    df["Handcap"] = pd.Categorical(df["Handcap"])
    handicap_dummies = pd.get_dummies(df["Handcap"], prefix="Handicap")
    df = pd.concat([df, handicap_dummies], axis=1)

    # Age filter
    df = df[(df.Age >= 0) & (df.Age <= 100)]

    # AwaitingTime
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])
    df["AwaitingTime"] = df["AppointmentDay"].sub(df["ScheduledDay"], axis=0)
    df["AwaitingTime"] = (df["AwaitingTime"] / np.timedelta64(1, "D")).abs()

    # Num_App_Missed (cumulative prior misses per patient, sorted by AppointmentDay)
    df_sorted = df.sort_values(by=["PatientId", "AppointmentDay"]).copy()
    df_sorted["Num_App_Missed"] = (
        df_sorted.groupby("PatientId")["No-show"]
        .transform(lambda x: x.shift().fillna(0).cumsum())
    )
    df["Num_App_Missed"] = df_sorted["Num_App_Missed"]

    # Drop unused cols
    df.drop(
        ["PatientId", "AppointmentID", "ScheduledDay", "Handcap",
         "AppointmentDay", "Neighbourhood"],
        axis=1, inplace=True
    )

    # Ensure all Handicap columns exist (0-4)
    for i in range(5):
        col = f"Handicap_{i}"
        if col not in df.columns:
            df[col] = 0

    # Feature matrix — exactly as in notebook
    feature_cols = [
        "Gender", "Diabetes", "Hipertension", "Scholarship", "SMS_received",
        "Handicap_0", "Handicap_1", "Handicap_2", "Handicap_3", "Handicap_4",
        "Num_App_Missed", "Age", "AwaitingTime"
    ]

    X = df[feature_cols]
    y = df["No-show"]

    # get_dummies (Gender: F→0, M→1 style columns)
    X = pd.get_dummies(X)

    return X, y


def train_and_save():
    print("📂 Loading data from:", CSV_PATH)
    X, y = load_and_preprocess(CSV_PATH)

    # Save ordered column list BEFORE scaling
    feature_cols_after_dummies = list(X.columns)
    joblib.dump(feature_cols_after_dummies, FEATURES_PATH)
    print(f"✅ Saved feature columns ({len(feature_cols_after_dummies)} features): {FEATURES_PATH}")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Saved scaler: {SCALER_PATH}")

    # Train/test split — exact from notebook (random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42
    )

    # Decision Tree — exact from notebook
    tree = DecisionTreeClassifier(max_depth=10, random_state=0)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)

    print(f"\n🌳 Decision Tree Results:")
    print(f"   Train Accuracy : {tree.score(X_train, y_train):.3f}")
    print(f"   Test  Accuracy : {tree.score(X_test, y_test):.3f}")
    print(f"   Precision      : {metrics.precision_score(y_test, y_pred):.3f}")
    print(f"   Recall         : {metrics.recall_score(y_test, y_pred):.3f}")

    joblib.dump(tree, MODEL_PATH)
    print(f"\n✅ Saved model: {MODEL_PATH}")
    print("\n🎯 Model brain ready. Run: streamlit run app.py")


if __name__ == "__main__":
    train_and_save()
