import pandas as pd

# Load Excel file
file_path = "H-1B_Data.xlsx"   # update path if needed
df = pd.read_excel(file_path)

# Inspect column names first (important)
print(df.columns)

# Replace with actual column names from your dataset
start_col = "APPLICATION_DATE"
end_col = "DECISION_DATE"

# Convert to datetime (handles messy formats safely)
df[start_col] = pd.to_datetime(df[start_col], errors='coerce')
df[end_col] = pd.to_datetime(df[end_col], errors='coerce')

# Calculate processing days
df["Processing Days"] = (df[end_col] - df[start_col]).dt.days

# Optional: handle negative or missing values
df["Processing Days"] = df["Processing Days"].apply(lambda x: x if x >= 0 else None)

# Save updated file
output_path = "H1B_with_processing_days.xlsx"
df.to_excel(output_path, index=False)

print("Processing days calculated and saved to:", output_path)