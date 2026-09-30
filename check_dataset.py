import pandas as pd

# Load dataset
df = pd.read_csv("data/patient_risk_dataset.csv")

# Show first 10 patients
print("FIRST 10 PATIENTS:")
print(df.head(10))

# Show all column names
print("\nCOLUMN NAMES:")
print(df.columns.tolist())

# Show dataset size
print("\nDATASET SHAPE:")
print(df.shape)

# Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Check risk distribution
print("\nRISK DISTRIBUTION:")
print(df["risk_level"].value_counts().sort_index())

# Show average age for each risk group
print("\nAVERAGE AGE BY RISK LEVEL:")
print(df.groupby("risk_level")["age"].mean())