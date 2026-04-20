import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- LOAD DATA ----------------
df = pd.read_excel("cleaned_H1B_data.xlsx")

# CLEAN COLUMN NAMES
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# ---------------- TARGET ----------------
TARGET = "processing_days"

if TARGET not in df.columns:
    raise ValueError("Target column not found")

# ---------------- DROP USELESS + HIGH CARDINALITY ----------------
drop_cols = [
    'case_number',
    'employer_name',
    'job_title',
    'soc_name',
    'worksite',
    'case_status',
    'agent_representing_employer'
]

df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

# ---------------- CLEAN TARGET ----------------
df = df[df[TARGET].notna()]
df = df[df[TARGET] >= 0]

# ---------------- SPLIT ----------------
X = df.drop(columns=[TARGET])
y = df[TARGET]

# ---------------- IDENTIFY TYPES ----------------
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object', 'string']).columns

# ---------------- DATA CLEANING ----------------
for col in cat_cols:
    X[col] = X[col].astype(str).fillna("missing")

for col in num_cols:
    X[col] = X[col].fillna(X[col].median())

# ---------------- PREPROCESSING ----------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
    ]
)

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- EVALUATION FUNCTION ----------------
def evaluate(y_true, y_pred, name):
    print(f"\n{name}")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R2:", r2_score(y_true, y_pred))

# ================= BASELINE MODELS =================
print("\n===== BASELINE MODELS =====")

models = {
    "Linear Regression": Pipeline([
        ('preprocessing', preprocessor),
        ('model', LinearRegression())
    ]),
    
    "Random Forest": Pipeline([
        ('preprocessing', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ]),
    
    "XGBoost": Pipeline([
        ('preprocessing', preprocessor),
        ('model', XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0))
    ])
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    evaluate(y_test, preds, name)

# ================= RANDOM FOREST TUNING =================
print("\n===== TUNING RANDOM FOREST =====")

rf_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', RandomForestRegressor(random_state=42, n_jobs=-1))
])

rf_params = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [10, 20, None],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2],
    'model__max_features': ['sqrt', 'log2']
}

rf_search = RandomizedSearchCV(
    rf_pipeline,
    rf_params,
    n_iter=10,
    cv=3,
    scoring='r2',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

rf_search.fit(X_train, y_train)

print("\nBest RF Params:", rf_search.best_params_)

rf_best = rf_search.best_estimator_
rf_preds = rf_best.predict(X_test)
evaluate(y_test, rf_preds, "Tuned Random Forest")

# ================= XGBOOST TUNING =================
print("\n===== TUNING XGBOOST =====")

xgb_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', XGBRegressor(random_state=42, n_jobs=-1, verbosity=0))
])

xgb_params = {
    'model__n_estimators': [200, 300],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__subsample': [0.7, 0.8],
    'model__colsample_bytree': [0.7, 0.8]
}

xgb_search = RandomizedSearchCV(
    xgb_pipeline,
    xgb_params,
    n_iter=10,
    cv=3,
    scoring='r2',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

xgb_search.fit(X_train, y_train)

print("\nBest XGB Params:", xgb_search.best_params_)

xgb_best = xgb_search.best_estimator_
xgb_preds = xgb_best.predict(X_test)
evaluate(y_test, xgb_preds, "Tuned XGBoost")

# ================= SAVE BEST MODEL =================
import joblib

joblib.dump(rf_best, "best_model.pkl")
print("\nBest model saved as best_model.pkl")