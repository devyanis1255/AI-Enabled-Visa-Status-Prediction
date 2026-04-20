import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def train_h1b_models(filepath):
    # 1. Load the processed data
    print(f"Reading {filepath}...")
    df = pd.read_csv(filepath)

    # 2. Feature Selection
    # We drop the target variable and any columns that are still 'object' (strings)
    # as ML models require purely numerical input.
    target = 'PROCESSING_TIME_DAYS'
    
    if target not in df.columns:
        print(f"Error: Target column '{target}' not found.")
        return

    # Drop non-numeric columns and the target to create feature set X
    X = df.select_dtypes(include=[np.number]).drop(columns=[target])
    y = df[target]

    print(f"Features used: {list(X.columns)}")
    
    # 3. Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- MODEL 1: LINEAR REGRESSION ---
    print("\n--- Training Linear Regression ---")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    evaluate_model("Linear Regression", y_test, lr_preds)

    # --- MODEL 2: RANDOM FOREST ---
    print("\n--- Training Random Forest ---")
    # n_jobs=-1 uses all your CPU cores for faster training
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    evaluate_model("Random Forest", y_test, rf_preds)

    # --- MODEL 3: XGBOOST ---
    print("\n--- Training XGBoost ---")
    # Requires: pip install xgboost
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    evaluate_model("XGBoost", y_test, xgb_preds)

def evaluate_model(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    
    print(f"Results for {name}:")
    print(f"  - Mean Absolute Error: {mae:.2f} days")
    print(f"  - Root Mean Squared Error: {rmse:.2f} days")
    print(f"  - R-Squared Score: {r2:.4f}")

if __name__ == "__main__":
    PROCESSED_DATA_PATH = 'H1B_Final_Processed_Data.csv'
    train_h1b_models(PROCESSED_DATA_PATH)