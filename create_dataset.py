import pandas as pd
import numpy as np

# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

np.random.seed(42)

N = 10000


# ============================================================
# 2. CREATE PATIENT DATA
# ============================================================

data = {

    "age": np.random.randint(1, 90, N),

    # Symptoms
    "fever": np.random.randint(0, 2, N),
    "cough": np.random.randint(0, 2, N),
    "breathing_difficulty": np.random.randint(0, 2, N),
    "chest_discomfort": np.random.randint(0, 2, N),
    "vomiting": np.random.randint(0, 2, N),
    "diarrhea": np.random.randint(0, 2, N),
    "dizziness": np.random.randint(0, 2, N),
    "fainting": np.random.randint(0, 2, N),
    "severe_weakness": np.random.randint(0, 2, N),

    # Actual duration in days
    "symptom_duration": np.random.randint(0, 31, N),

    # Medical history
    "diabetes": np.random.randint(0, 2, N),
    "hypertension": np.random.randint(0, 2, N),
    "heart_disease": np.random.randint(0, 2, N),
    "asthma": np.random.randint(0, 2, N)
}


df = pd.DataFrame(data)


# ============================================================
# 3. CREATE SYNTHETIC RISK SCORE
# ============================================================

def calculate_risk_score(row):

    score = 0

    # Strong warning symptoms
    score += row["breathing_difficulty"] * 4
    score += row["fainting"] * 4
    score += row["chest_discomfort"] * 3

    # Other symptoms
    score += row["fever"] * 1
    score += row["cough"] * 1
    score += row["vomiting"] * 1
    score += row["diarrhea"] * 1
    score += row["dizziness"] * 2
    score += row["severe_weakness"] * 2

    # Symptom duration
    if row["symptom_duration"] >= 7:
        score += 2
    elif row["symptom_duration"] >= 3:
        score += 1

    # Medical history
    score += row["diabetes"] * 1
    score += row["hypertension"] * 1
    score += row["heart_disease"] * 2
    score += row["asthma"] * 1

    # Age
    if row["age"] >= 65:
        score += 1

    return score


df["synthetic_risk_score"] = df.apply(
    calculate_risk_score,
    axis=1
)


# ============================================================
# 4. CREATE RISK CLASSES
# ============================================================

# Add a small amount of uncertainty so the synthetic
# dataset does not behave like a perfect deterministic rule.

noise = np.random.normal(
    loc=0,
    scale=2.0,
    size=N
)

risk_signal = (
    df["synthetic_risk_score"] + noise
)


# Use percentile thresholds
low_threshold = np.percentile(risk_signal, 40)
high_threshold = np.percentile(risk_signal, 70)


df["risk_level"] = np.select(
    [
        risk_signal < low_threshold,
        risk_signal < high_threshold
    ],
    [
        0,
        1
    ],
    default=2
)


# ============================================================
# 5. REMOVE INTERNAL SYNTHETIC SCORE
# ============================================================

# This score is only used to create the synthetic target.
# It should NOT be given to the ML model.

df = df.drop(
    "synthetic_risk_score",
    axis=1
)


# ============================================================
# 6. CHECK DATASET
# ============================================================

print("\n========================================")
print("DATASET")
print("========================================")

print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


print("\nRisk distribution:")

print(
    df["risk_level"]
    .value_counts()
    .sort_index()
)


print("\nRisk percentages:")

print(
    df["risk_level"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)


# ============================================================
# 7. SAVE
# ============================================================

df.to_csv(
    "data/patient_risk_dataset.csv",
    index=False
)

print("\nDataset saved successfully.")

print(
    "data/patient_risk_dataset.csv"
)