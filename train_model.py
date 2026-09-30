import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report


# ==============================
# 1. LOAD DATASET
# ==============================

data = pd.read_csv("data/patient_risk_dataset.csv")

print("Dataset loaded!")
print("Shape:", data.shape)


# ==============================
# 2. SEPARATE FEATURES & TARGET
# ==============================

X = data.drop("risk_level", axis=1)
y = data["risk_level"]


# ==============================
# 3. TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==============================
# 4. CREATE MODEL
# ==============================

model = GaussianNB()


# ==============================
# 5. TRAIN MODEL
# ==============================

model.fit(X_train, y_train)

print("\nModel training completed!")


# ==============================
# 6. PREDICTION
# ==============================

y_pred = model.predict(X_test)


# ==============================
# 7. EVALUATION
# ==============================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==============================
# 8. SAVE MODEL
# ==============================

joblib.dump(model, "rural_health_model.pkl")

print("\nModel saved as:")
print("rural_health_model.pkl")