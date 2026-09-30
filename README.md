# RuralHealthAI 🏥🤖

RuralHealthAI is an AI-assisted healthcare decision-support prototype designed to help people in rural areas perform an initial symptom-based health-risk assessment and connect with available healthcare workers.

## 🎯 Problem Statement

People in rural areas may face delays in accessing healthcare due to distance, limited healthcare facilities, lack of immediate medical guidance, and communication barriers.

RuralHealthAI aims to provide a simple digital platform that can:

- Collect basic patient information and symptoms
- Perform an AI-based risk classification
- Classify the assessment into Low, Moderate, or High risk
- Display the estimated risk probabilities
- Help connect the patient with a healthcare worker
- Provide healthcare-worker assistance for further evaluation

## 🧠 AI / Machine Learning

The project includes a machine-learning based patient risk classification module.

### Input Features

The model uses patient information such as:

- Age
- Fever
- Cough
- Breathing difficulty
- Chest discomfort
- Vomiting
- Diarrhea
- Dizziness
- Fainting
- Severe weakness
- Symptom duration
- Diabetes
- Hypertension
- Heart disease
- Asthma

### Risk Categories

The system produces three risk categories:

| Risk Level | Description |
|---|---|
| 🟢 Low | Lower-risk symptom pattern |
| 🟡 Moderate | Moderate-risk symptom pattern |
| 🔴 High | Higher-risk symptom pattern |

## 🏗️ Project Structure

```text
RuralHealthAI/
│
├── backend/
│
├── data/
│   └── patient_risk_dataset.csv
│
├── frontend/
│   ├── index.html
│   ├── assessment.html
│   ├── result.html
│   └── nurse.html
│
├── app.py
├── create_dataset.py
├── check_dataset.py
├── train_model.py
├── rural_health_model.pkl
├── joint.pdf
└── README.md