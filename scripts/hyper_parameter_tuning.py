# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, r2_score
# from sklearn.preprocessing import LabelEncoder

# def train_realistic_model(filepath):
#     df = pd.read_csv(filepath, low_memory=False)
    
#     # 1. Drop columns that are "Cheating" (the answer is hidden in these)
#     # Also drop date columns because they contain the answer
#     to_drop = ['PROCESSING_TIME_DAYS', 'Unnamed: 0', 'APPLICATION_DATE', 'DECISION_DATE', 'CASE_NUMBER']
    
#     # 2. Encode categorical columns (The "Secret Sauce")
#     # This turns 'NEW YORK' into 1, 'CALIFORNIA' into 2, etc.
#     le = LabelEncoder()
#     categorical_cols = df.select_dtypes(include=['object']).columns
    
#     for col in categorical_cols:
#         if col not in to_drop:
#             df[col] = le.fit_transform(df[col].astype(str))

#     # 3. Define Features and Target
#     X = df.drop(columns=[c for c in to_drop if c in df.columns], errors='ignore')
#     X = X.select_dtypes(include=[np.number]) # Ensure only numbers remain
#     y = df['PROCESSING_TIME_DAYS']
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     print(f"Training on {X.shape[1]} features including Encoded Categoricals...")

#     # 4. Train Random Forest
#     model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
#     model.fit(X_train, y_train)
    
#     # 5. Evaluate
#     preds = model.predict(X_test)
#     print("\n--- Realistic Model Results ---")
#     print(f"MAE: {mean_absolute_error(y_test, preds):.2f} days")
#     print(f"R-Squared: {r2_score(y_test, preds):.4f}")

#     return model

# if __name__ == "__main__":
#     train_realistic_model('H1B_Final_Processed_Data.csv')

#OUTPUT
#--- Realistic Model Results ---
#MAE: 59.98 days
#R-Squared: 0.3845

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def tune_random_forest(filepath):
    # 1. Load and Prepare Data
    # low_memory=False prevents the DtypeWarning you saw earlier
    df = pd.read_csv(filepath, low_memory=False)
    target = 'PROCESSING_TIME_DAYS'
    
    # Preprocessing: Drop target and the useless 'Unnamed: 0' index if it exists
    X = df.select_dtypes(include=[np.number]).drop(columns=[target, 'Unnamed: 0'], errors='ignore')
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Define the Parameter Grid
    param_grid = {
        'n_estimators': [100, 200],      # Number of trees
        'max_depth': [10, 20, None],     # Tree depth
        'min_samples_split': [2, 5],     # Samples needed to split
        'max_features': ['sqrt', 'log2'] # Features to consider per split
    }

    print("Starting Grid Search... (Using all CPU cores)")
    
    # 3. Initialize GridSearchCV
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=3, 
        scoring='r2', 
        verbose=1, 
        n_jobs=-1
    )

    # 4. Fit the model
    grid_search.fit(X_train, y_train)

    # 5. Results
    print("\n--- Best Parameters Found ---")
    print(grid_search.best_params_)

    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)

    print("\n--- Performance of Tuned Model ---")
    print(f"MAE: {mean_absolute_error(y_test, predictions):.2f} days")
    print(f"R-Squared: {r2_score(y_test, predictions):.4f}")

    # 6. Feature Importance (Bonus)
    # This shows you which columns actually mattered most
    importances = best_model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    print("\n--- Top 5 Most Important Features ---")
    print(feature_importance_df.head(5))

    return best_model

if __name__ == "__main__":
    PROCESSED_DATA_PATH = 'H1B_Final_Processed_Data.csv'
    best_rf_model = tune_random_forest(PROCESSED_DATA_PATH)



# OUTPUT
# --- Best Parameters Found ---
# {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_split': 5, 'n_estimators': 200}

# --- Performance of Tuned Model ---
# MAE: 75.89 days
# R-Squared: 0.0635

# --- Top 5 Most Important Features ---
#             Feature  Importance
# 9   PREVAILING_WAGE    0.599266
# 0        NAICS_CODE    0.254140
# 2    NEW_EMPLOYMENT    0.063673
# 7  AMENDED_PETITION    0.021586
# 1     TOTAL_WORKERS    0.019240
