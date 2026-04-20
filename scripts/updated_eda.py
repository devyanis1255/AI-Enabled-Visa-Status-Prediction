import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------- CONFIG ---------------- #

INPUT_FILE = "updated_cleaned_H1B_data.xlsx"
OUTPUT_FILE = "updated_H1B_EDA_Output.xlsx"

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 4)

# ---------------- EDA FUNCTION ---------------- #

def perform_eda(file):

    if not os.path.exists(file):
        print("File not found.")
        return

    print("Loading dataset...")
    df = pd.read_excel(INPUT_FILE)
    df.columns = df.columns.str.strip().str.upper()

    print("\nDataset Shape:", df.shape)
    print("\nMissing Values (Top 15):")
    print(df.isnull().sum().sort_values(ascending=False).head(15))
    print("\nDuplicate Rows:", df.duplicated().sum())

    # ---------------- TARGET ANALYSIS ---------------- #

    if "CASE_STATUS" in df.columns:
        print("\nTarget Distribution:")
        print(df["CASE_STATUS"].value_counts(normalize=True))

        plt.figure()
        sns.countplot(
            y="CASE_STATUS",
            data=df,
            order=df["CASE_STATUS"].value_counts().index
        )
        plt.title("CASE_STATUS Distribution")
        plt.tight_layout()
        plt.show()

    # ---------------- WAGE ANALYSIS ---------------- #

    def clean_currency(value):
        if isinstance(value, str):
            value = value.replace("$", "").replace(",", "")
        return pd.to_numeric(value, errors="coerce")

    if "PREVAILING_WAGE" in df.columns:

        df["PREVAILING_WAGE"] = df["PREVAILING_WAGE"].apply(clean_currency)

        plt.figure()
        sns.histplot(df["PREVAILING_WAGE"].dropna(), bins=50)
        plt.title("Prevailing Wage Distribution")
        plt.tight_layout()
        plt.show()

        df["LOG_WAGE"] = np.log1p(df["PREVAILING_WAGE"])

        plt.figure()
        sns.histplot(df["LOG_WAGE"].dropna(), bins=50, kde=True)
        plt.title("Log Transformed Wage Distribution")
        plt.tight_layout()
        plt.show()

        if "CASE_STATUS" in df.columns:
            plt.figure()
            sns.boxplot(x="CASE_STATUS", y="LOG_WAGE", data=df)
            plt.title("Wage vs Case Status")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    # ---------------- EMPLOYER ANALYSIS ---------------- #

    if "EMPLOYER_NAME" in df.columns:
        top_employers = df["EMPLOYER_NAME"].value_counts().head(10)

        plt.figure()
        sns.barplot(
            x=top_employers.values,
            y=top_employers.index
        )
        plt.title("Top 10 Employers")
        plt.tight_layout()
        plt.show()

    # ---------------- STATE ANALYSIS ---------------- #

    if "WORKSITE_STATE" in df.columns:
        top_states = df["WORKSITE_STATE"].value_counts().head(10)

        plt.figure()
        sns.barplot(
            x=top_states.values,
            y=top_states.index
        )
        plt.title("Top 10 Worksite States")
        plt.tight_layout()
        plt.show()

        if "CASE_STATUS" in df.columns:
            state_status = (
                df.groupby("WORKSITE_STATE")["CASE_STATUS"]
                .value_counts(normalize=True)
                .unstack()
                .fillna(0)
            )

            state_status.head(10).plot(
                kind="bar",
                stacked=True,
                figsize=(10, 5)
            )
            plt.title("State-wise Case Status Distribution")
            plt.ylabel("Proportion")
            plt.tight_layout()
            plt.show()

    # ---------------- JOB TITLE ANALYSIS ---------------- #

    if "JOB_TITLE" in df.columns:
        top_jobs = df["JOB_TITLE"].value_counts().head(10)

        plt.figure()
        sns.barplot(
            x=top_jobs.values,
            y=top_jobs.index
        )
        plt.title("Top 10 Job Titles")
        plt.tight_layout()
        plt.show()

    # ---------------- YEAR ANALYSIS ---------------- #

    if "CASE_SUBMITTED" in df.columns:
        df["CASE_SUBMITTED"] = pd.to_datetime(
            df["CASE_SUBMITTED"], errors="coerce"
        )
        df["APPLICATION_YEAR"] = df["CASE_SUBMITTED"].dt.year

        plt.figure()
        sns.countplot(x="APPLICATION_YEAR", data=df)
        plt.title("Applications Per Year")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # ---------------- CORRELATION ---------------- #

    numeric_df = df.select_dtypes(include=np.number)

    if not numeric_df.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            numeric_df.corr(),
            cmap="coolwarm",
            annot=False
        )
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.show()

    # ---------------- SAVE PROCESSED DATA ---------------- #

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nProcessed dataset saved as: {OUTPUT_FILE}")
    print("EDA Completed Successfully.")


# ---------------- RUN SCRIPT ---------------- #

if __name__ == "__main__":
    perform_eda(INPUT_FILE)