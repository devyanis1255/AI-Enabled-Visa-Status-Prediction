import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- LOAD ----------------
df = pd.read_excel("cleaned_H1B_data.xlsx")

# CLEAN COLUMN NAMES
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# ---------------- TARGET ----------------
TARGET = "processing_days"

# ---------------- DROP BAD COLUMNS ----------------
df = df.drop(columns=[
    'sr._no.', 'case_number', 'employer_address'
], errors='ignore')

# ---------------- DATE FEATURES ----------------
df['application_date'] = pd.to_datetime(df['application_date'], errors='coerce')
df['decision_date'] = pd.to_datetime(df['decision_date'], errors='coerce')

df['app_year'] = df['application_date'].dt.year
df['app_month'] = df['application_date'].dt.month
df['dec_year'] = df['decision_date'].dt.year
df['dec_month'] = df['decision_date'].dt.month

df = df.drop(columns=['application_date', 'decision_date'], errors='ignore')

# ---------------- FEATURE ENGINEERING ----------------
df['log_wage'] = np.log1p(df['prevailing_wage'])
df = df.drop(columns=['prevailing_wage'], errors='ignore')

# ---------------- ENCODE ----------------
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    df[col] = df[col].astype(str)
    df[col] = df[col].map(df[col].value_counts())

# ---------------- SPLIT ----------------
X = df.drop(columns=[TARGET])
y = df[TARGET]

X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODELS ----------------
models = {
    "Linear": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=100, n_jobs=-1)
}

# ---------------- TRAIN + EVALUATE ----------------
best_model = None
best_score = -999

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    print(f"{name} R2:", r2)

    if r2 > best_score:
        best_score = r2
        best_model = model

# ---------------- SAVE ----------------
joblib.dump(best_model, "best_model.pkl")
joblib.dump(list(X.columns), "columns.pkl")

print("\n✅ Best model saved")