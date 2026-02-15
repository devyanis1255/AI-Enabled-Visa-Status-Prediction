import pandas as pd
import os
if __name__ == "__main__":
    # If your file is named 'training_data.xlsx', change it here:
    scrub_data('H-1B_Disclosure-data.xlsx')

def scrub_data(file_path):
    # 1. Load the dataset
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return

    print(f"Loading {file_path}...")
    df = pd.read_excel(file_path)

    # 2. Handling Duplicates
    initial_count = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Removed {initial_count - len(df)} duplicate rows.")

    # 3. Handling Missing Values
    # Fill numeric columns with Median, categorical with 'Unknown'
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(df[col].median())
    print("Handled missing values (Imputation complete).")

    # 4. Save the cleaned version
    output_file = "cleaned_data.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Success! Cleaned data saved as: {output_file}")

if __name__ == "__main__":
    # Change 'dataset.xlsx' to your actual file name
    scrub_data('dataset.xlsx')