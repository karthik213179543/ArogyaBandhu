from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# ============================================================
# RURALHEALTHAI BACKEND
# ============================================================

app = FastAPI(
    title="RuralHealthAI API",
    description="AI-powered rural healthcare risk assessment",
    version="1.0.0"
)


# ============================================================
# LOAD TRAINED ML MODEL
# ============================================================

model = joblib.load("backend/rural_health_model.pkl")


# ============================================================
# PATIENT DATA
# ============================================================

class PatientData(BaseModel):
    age: int
    fever: int
    cough: int
    breathing_difficulty: int
    chest_discomfort: int
    vomiting: int
    diarrhea: int
    dizziness: int
    fainting: int
    severe_weakness: int
    symptom_duration: int
    diabetes: int
    hypertension: int
    heart_disease: int
    asthma: int


# ============================================================
# HOME / HEALTH CHECK
# ============================================================

@app.get("/")
def home():
    return {
        "message": "RuralHealthAI API is running",
        "status": "success"
    }


# ============================================================
# ML PREDICTION
# ============================================================

@app.post("/predict")
def predict_risk(patient: PatientData):

    input_data = pd.DataFrame([{
        "age": patient.age,
        "fever": patient.fever,
        "cough": patient.cough,
        "breathing_difficulty": patient.breathing_difficulty,
        "chest_discomfort": patient.chest_discomfort,
        "vomiting": patient.vomiting,
        "diarrhea": patient.diarrhea,
        "dizziness": patient.dizziness,
        "fainting": patient.fainting,
        "severe_weakness": patient.severe_weakness,
        "symptom_duration": patient.symptom_duration,
        "diabetes": patient.diabetes,
        "hypertension": patient.hypertension,
        "heart_disease": patient.heart_disease,
        "asthma": patient.asthma
    }])

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    classes = list(model.classes_)

    probability_map = dict(
        zip(classes, probabilities)
    )

    risk_names = {
        0: "LOW",
        1: "MODERATE",
        2: "HIGH"
    }

    return {
        "risk_level": risk_names.get(
            int(prediction),
            "UNKNOWN"
        ),
        "risk_code": int(prediction),
        "probabilities": {
            "low": round(
                probability_map.get(0, 0) * 100,
                2
            ),
            "moderate": round(
                probability_map.get(1, 0) * 100,
                2
            ),
            "high": round(
                probability_map.get(2, 0) * 100,
                2
            )
        }
    }