import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# --- CONFIGURATION ---
INPUT_FILE = 'H-1B_Disclosure_Data.xlsx'
OUTPUT_FILE = 'H1B_Final_Processed_Data.xlsx'

def process_h1b_data(filename):
    if not os.path.exists(filename):
        print(f"❌ Error: {filename} not found in the current directory.")
        return

    print("⏳ Loading dataset...")
    df = pd.read_excel(filename)
    
    # 1. Standardize Column Names & Types
    # Handle the CASE_SUBMITTED to APPLICATION_DATE renaming
    if 'CASE_SUBMITTED' in df.columns:
        df.rename(columns={'CASE_SUBMITTED': 'APPLICATION_DATE'}, inplace=True)
    
    # 2. Date Processing & Feature Engineering
    print("📅 Calculating processing times...")
    df['APPLICATION_DATE'] = pd.to_datetime(df['APPLICATION_DATE'], errors='coerce')
    df['DECISION_DATE'] = pd.to_datetime(df['DECISION_DATE'], errors='coerce')
    
    # Calculate days; fill missing results with the median
    df['PROCESSING_TIME_DAYS'] = (df['DECISION_DATE'] - df['APPLICATION_DATE']).dt.days
    df['PROCESSING_TIME_DAYS'] = df['PROCESSING_TIME_DAYS'].fillna(df['PROCESSING_TIME_DAYS'].median())

    # 3. Text Normalization
    print("🔠 Normalizing text columns...")
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip().str.upper()

    # 4. Wage Cleaning (Numeric Conversion)
    def clean_currency(value):
        if isinstance(value, str):
            return pd.to_numeric(value.replace('$', '').replace(',', ''), errors='coerce')
        return value

    wage_cols = ['PREVAILING_WAGE', 'WAGE_RATE_OF_PAY_FROM']
    for col in wage_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_currency)

    # 5. Handle Missing Values (Imputation)
    print("🧹 Filling missing values...")
    # Fill Numeric with Median
    num_vars = df.select_dtypes(include=[np.number]).columns
    df[num_vars] = df[num_vars].fillna(df[num_vars].median())
    
    # Fill Categorical with Mode (most frequent)
    for col in text_cols:
        if not df[col].empty:
            df[col] = df[col].replace('NAN', 'UNKNOWN').fillna('UNKNOWN')

    # 6. Machine Learning Encoding
    print("🤖 Encoding categorical variables...")
    # Label Encode binary-like columns
    if 'FULL_TIME_POSITION' in df.columns:
        le = LabelEncoder()
        df['FULL_TIME_POSITION'] = le.fit_transform(df['FULL_TIME_POSITION'].astype(str))

    # One-Hot Encode Case Status
    if 'CASE_STATUS' in df.columns:
        df = pd.get_dummies(df, columns=['CASE_STATUS'], drop_first=True)

    # 7. Final Export
    print(f"💾 Saving cleaned data to {OUTPUT_FILE}...")
    df.to_excel(OUTPUT_FILE, index=False)
    print("✨ SUCCESS: Process Complete!")
    print(df.head())

if __name__ == "__main__":
    process_h1b_data(INPUT_FILE)