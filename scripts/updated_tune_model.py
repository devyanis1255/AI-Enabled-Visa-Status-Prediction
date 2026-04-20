import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# LOAD DATA
df = pd.read_excel("cleaned_H1B_data.xlsx")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

TARGET = "processing_days"

# SAME PREPROCESSING (COPY EXACTLY)
df = df.drop(columns=['sr._no.', 'case_number', 'employer_address'], errors='ignore')

df['application_date'] = pd.to_datetime(df['application_date'], errors='coerce')
df['decision_date'] = pd.to_datetime(df['decision_date'], errors='coerce')

df['app_year'] = df['application_date'].dt.year
df['app_month'] = df['application_date'].dt.month
df['dec_year'] = df['decision_date'].dt.year
df['dec_month'] = df['decision_date'].dt.month

df = df.drop(columns=['application_date', 'decision_date'], errors='ignore')

df['log_wage'] = np.log1p(df['prevailing_wage'])
df = df.drop(columns=['prevailing_wage'], errors='ignore')

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].map(df[col].value_counts())

X = df.drop(columns=[TARGET]).fillna(0)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y)

# ---------------- TUNE XGBOOST ----------------
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 1.0]
}

model = XGBRegressor(n_jobs=-1)

search = RandomizedSearchCV(
    model,
    param_grid,
    n_iter=10,
    cv=3,
    scoring='r2',
    verbose=1
)

search.fit(X_train, y_train)

best_model = search.best_estimator_

joblib.dump(best_model, "best_model.pkl")

print("✅ Tuned model saved")