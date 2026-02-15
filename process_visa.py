import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_FILE = 'H-1B_Data.csv'
OUTPUT_FILE = 'H1B_Final_Processed_Data.csv'

def process_h1b_data(filename):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found in the current directory.")
        return

    print("Loading dataset...")
    df = pd.read_csv(filename)

    # 1. Standardize Column Names
    if 'CASE_SUBMITTED' in df.columns:
        df.rename(columns={'CASE_SUBMITTED': 'APPLICATION_DATE'}, inplace=True)

    # 2. Date Processing
    print("Calculating processing times...")
    df['APPLICATION_DATE'] = pd.to_datetime(df.get('APPLICATION_DATE'), errors='coerce')
    df['DECISION_DATE'] = pd.to_datetime(df.get('DECISION_DATE'), errors='coerce')

    df['PROCESSING_TIME_DAYS'] = (
        df['DECISION_DATE'] - df['APPLICATION_DATE']
    ).dt.days

    df['PROCESSING_TIME_DAYS'] = df['PROCESSING_TIME_DAYS'].fillna(
        df['PROCESSING_TIME_DAYS'].median()
    )

    # 3. Text Normalization
    print("Normalizing text columns...")
    text_cols = df.select_dtypes(include='object').columns
    for col in text_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace('NAN', 'UNKNOWN')
        )

    # 4. Wage Cleaning
    def clean_currency(value):
        if isinstance(value, str):
            value = value.replace('$', '').replace(',', '')
        return pd.to_numeric(value, errors='coerce')

    wage_cols = ['PREVAILING_WAGE', 'WAGE_RATE_OF_PAY_FROM']
    for col in wage_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_currency)

    # 5. Missing Value Handling
    print("Filling missing values...")
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    df[text_cols] = df[text_cols].fillna('UNKNOWN')

    # 6. Encoding (NO sklearn)
    print("Encoding categorical variables...")

    # Binary encoding for FULL_TIME_POSITION
    if 'FULL_TIME_POSITION' in df.columns:
        df['FULL_TIME_POSITION'] = (
            df['FULL_TIME_POSITION']
            .map({'Y': 1, 'YES': 1, 'TRUE': 1,
                  'N': 0, 'NO': 0, 'FALSE': 0})
            .fillna(0)
            .astype(int)
        )

    # One-Hot Encoding for CASE_STATUS
    if 'CASE_STATUS' in df.columns:
        df = pd.get_dummies(df, columns=['CASE_STATUS'], drop_first=True)

    # 7. Export
    print(f"Saving processed data to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)

    print("SUCCESS: Processing completed.")
    print(df.head())

if __name__ == "__main__":
    process_h1b_data(INPUT_FILE)
