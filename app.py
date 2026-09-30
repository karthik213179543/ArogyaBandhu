import streamlit as st

import pandas as pd

import joblib





# ============================================================

# PAGE CONFIGURATION

# ============================================================



st.set_page_config(

    page_title="RuralHealthAI",

    page_icon="🏥",

    layout="wide"

)





# ============================================================

# LOAD MODEL

# ============================================================



@st.cache_resource

def load_model():

    return joblib.load("rural_health_model.pkl")





model = load_model()





# ============================================================

# HEADER

# ============================================================



st.title("🏥 RuralHealthAI")

st.subheader("AI-Powered Rural Patient Risk Assessment")



st.write(

    "A machine-learning decision-support prototype designed to "

    "help identify patients who may require closer attention."

)



st.divider()





# ============================================================

# PATIENT INFORMATION

# ============================================================



st.header("👤 Patient Information")



col1, col2 = st.columns(2)



with col1:

    age = st.number_input(

        "Age",

        min_value=1,

        max_value=120,

        value=30

    )



with col2:

    symptom_duration = st.number_input(

        "Symptom Duration (days)",

        min_value=0,

        max_value=30,

        value=2

    )





# ============================================================

# SYMPTOMS

# ============================================================



st.header("🩺 Symptoms")

st.caption("Select Yes if the patient currently has the symptom.")



col1, col2, col3, col4 = st.columns(4)



with col1:

    fever = st.selectbox("Fever", ["No", "Yes"])



with col2:

    cough = st.selectbox("Cough", ["No", "Yes"])



with col3:

    breathing_difficulty = st.selectbox(

        "Breathing Difficulty",

        ["No", "Yes"]

    )



with col4:

    chest_discomfort = st.selectbox(

        "Chest Discomfort",

        ["No", "Yes"]

    )





col1, col2, col3, col4 = st.columns(4)



with col1:

    vomiting = st.selectbox("Vomiting", ["No", "Yes"])



with col2:

    diarrhea = st.selectbox("Diarrhea", ["No", "Yes"])



with col3:

    dizziness = st.selectbox("Dizziness", ["No", "Yes"])



with col4:

    fainting = st.selectbox("Fainting", ["No", "Yes"])





col1, col2 = st.columns(2)



with col1:

    severe_weakness = st.selectbox(

        "Severe Weakness",

        ["No", "Yes"]

    )



with col2:

    severity_score = st.slider(

        "Overall Severity Score",

        min_value=0,

        max_value=10,

        value=3

    )





# ============================================================

# MEDICAL HISTORY

# ============================================================



st.header("📋 Medical History")

st.caption("Select Yes if the patient has the condition.")



col1, col2, col3, col4 = st.columns(4)



with col1:

    diabetes = st.selectbox("Diabetes", ["No", "Yes"])



with col2:

    hypertension = st.selectbox("Hypertension", ["No", "Yes"])



with col3:

    heart_disease = st.selectbox(

        "Heart Disease",

        ["No", "Yes"]

    )



with col4:

    asthma = st.selectbox("Asthma", ["No", "Yes"])





st.divider()





# ============================================================

# HELPER FUNCTION

# ============================================================



def yes_no(value):

    return 1 if value == "Yes" else 0





# ============================================================

# MODEL FEATURE NAMES

# ============================================================



feature_names = [

    "age",

    "fever",

    "cough",

    "breathing_difficulty",

    "chest_discomfort",

    "vomiting",

    "diarrhea",

    "dizziness",

    "fainting",

    "severe_weakness",

    "symptom_duration",

    "diabetes",

    "hypertension",

    "heart_disease",

    "asthma",


]



display_names = {

    "age": "Age",

    "fever": "Fever",

    "cough": "Cough",

    "breathing_difficulty": "Breathing Difficulty",

    "chest_discomfort": "Chest Discomfort",

    "vomiting": "Vomiting",

    "diarrhea": "Diarrhea",

    "dizziness": "Dizziness",

    "fainting": "Fainting",

    "severe_weakness": "Severe Weakness",

    "symptom_duration": "Symptom Duration",

    "diabetes": "Diabetes",

    "hypertension": "Hypertension",

    "heart_disease": "Heart Disease",

    "asthma": "Asthma",


}





# ============================================================

# MODEL EXPLANATION

# ============================================================



def get_feature_importance(model, columns):

    """

    Get global feature importance from tree-based models.

    This describes which features were important during training;

    it does not claim that a feature caused an individual prediction.

    """



    if not hasattr(model, "feature_importances_"):

        return None



    importances = model.feature_importances_



    if len(importances) != len(columns):

        return None



    importance_df = pd.DataFrame({

        "Feature": [display_names.get(c, c) for c in columns],

        "Importance": importances

    })



    return importance_df.sort_values(

        "Importance",

        ascending=False

    ).reset_index(drop=True)





# ============================================================

# PREDICTION

# ============================================================



if st.button(

    "🔍 Assess Patient Risk",

    use_container_width=True,

    type="primary"

):



    # --------------------------------------------------------

    # CREATE MODEL INPUT

    # --------------------------------------------------------



    input_data = pd.DataFrame([{

        "age": age,

        "fever": yes_no(fever),

        "cough": yes_no(cough),

        "breathing_difficulty": yes_no(breathing_difficulty),

        "chest_discomfort": yes_no(chest_discomfort),

        "vomiting": yes_no(vomiting),

        "diarrhea": yes_no(diarrhea),

        "dizziness": yes_no(dizziness),

        "fainting": yes_no(fainting),

        "severe_weakness": yes_no(severe_weakness),

        "symptom_duration": symptom_duration,

        "diabetes": yes_no(diabetes),

        "hypertension": yes_no(hypertension),

        "heart_disease": yes_no(heart_disease),

        "asthma": yes_no(asthma),


    }])





    # --------------------------------------------------------

    # MODEL PREDICTION

    # --------------------------------------------------------



    prediction = model.predict(input_data)[0]



    probabilities = model.predict_proba(input_data)[0]



    classes = list(model.classes_)



    probability_map = dict(zip(classes, probabilities))



    risk_names = {

        0: "LOW RISK",

        1: "MEDIUM RISK",

        2: "HIGH RISK"

    }



    risk_name = risk_names.get(

        prediction,

        f"RISK CLASS {prediction}"

    )





    # ========================================================

    # AI ASSESSMENT

    # ========================================================



    st.divider()

    st.header("🧠 AI Assessment")



    if prediction == 0:



        st.success(f"### 🟢 {risk_name}")



        st.write(

            "For this particular input, the trained model assigns "

            "the highest predicted probability to the Low Risk class. "

            "This is a model output and not a medical diagnosis."

        )



    elif prediction == 1:



        st.warning(f"### 🟡 {risk_name}")



        st.write(

            "For this particular input, the trained model assigns "

            "the highest predicted probability to the Medium Risk class. "

            "This is a model output and not a medical diagnosis."

        )



    else:



        st.error(f"### 🔴 {risk_name}")



        st.write(

            "For this particular input, the trained model assigns "

            "the highest predicted probability to the High Risk class. "

            "Professional healthcare evaluation should be considered. "

            "This is a model output and not a medical diagnosis."

        )





    # ========================================================

    # MODEL CONFIDENCE

    # ========================================================



    st.subheader("📊 Model Confidence")



    p1, p2, p3 = st.columns(3)



    with p1:

        st.metric(

            "🟢 Low Risk",

            f"{probability_map.get(0, 0) * 100:.1f}%"

        )



    with p2:

        st.metric(

            "🟡 Medium Risk",

            f"{probability_map.get(1, 0) * 100:.1f}%"

        )



    with p3:

        st.metric(

            "🔴 High Risk",

            f"{probability_map.get(2, 0) * 100:.1f}%"

        )





    # ========================================================

    # IMPROVED RISK PROBABILITY

    # ========================================================



    st.subheader("📈 Risk Probability")



    risk_probabilities = [

        ("🟢 Low Risk", probability_map.get(0, 0)),

        ("🟡 Medium Risk", probability_map.get(1, 0)),

        ("🔴 High Risk", probability_map.get(2, 0))

    ]



    for risk_label, probability in risk_probabilities:



        percentage = probability * 100



        st.write(

            f"**{risk_label} — {percentage:.1f}%**"

        )



        st.progress(

            int(round(percentage))

        )





    # ========================================================

    # KEY FACTORS CONSIDERED

    # ========================================================



    st.divider()



    st.header("🩺 Key Factors Considered")



    st.write(

        "These are the patient inputs provided to the model for this "

        "prediction. They describe the information considered by the "

        "prototype and do not represent a medical diagnosis."

    )



    factor_items = [

        ("🌡️ Fever", fever),

        ("😷 Cough", cough),

        ("🫁 Breathing Difficulty", breathing_difficulty),

        ("❤️ Chest Discomfort", chest_discomfort),

        ("🤢 Vomiting", vomiting),

        ("💧 Diarrhea", diarrhea),

        ("🧠 Dizziness", dizziness),

        ("⚠️ Fainting", fainting),

        ("💪 Severe Weakness", severe_weakness),

        ("📅 Symptom Duration", f"{symptom_duration} days"),

    ]



    history_items = [

        ("Diabetes", diabetes),

        ("Hypertension", hypertension),

        ("Heart Disease", heart_disease),

        ("Asthma", asthma),

    ]



    factor_cols = st.columns(2)



    with factor_cols[0]:

        for label, value in factor_items[:5]:

            if isinstance(value, str):

                st.write(f"**{label}:** {value}")

            else:

                st.write(f"**{label}:** {'Yes' if value == 'Yes' else 'No'}")



    with factor_cols[1]:

        for label, value in factor_items[5:]:

            if isinstance(value, str):

                st.write(f"**{label}:** {value}")

            else:

                st.write(f"**{label}:** {'Yes' if value == 'Yes' else 'No'}")



    st.caption(

        "Medical history: "

        + ", ".join(

            f"{name}: {'Yes' if value == 'Yes' else 'No'}"

            for name, value in history_items

        )

    )



    # ========================================================

    # EMERGENCY SYMPTOM WARNING

    # ========================================================



    emergency_symptoms = []



    if breathing_difficulty == "Yes":

        emergency_symptoms.append("Breathing Difficulty")

    if chest_discomfort == "Yes":

        emergency_symptoms.append("Chest Discomfort")

    if fainting == "Yes":

        emergency_symptoms.append("Fainting")



    if emergency_symptoms:

        st.error(

            "🚨 **Important Symptom Alert**\n\n"

            "The patient reported: **"

            + ", ".join(emergency_symptoms)

            + "**. These symptoms can require prompt medical evaluation. "

            "Do not rely on the AI risk score to decide whether care is needed."

        )



    # WHY DID THE MODEL PREDICT THIS?

    # ========================================================



    st.divider()



    st.header("🔎 Why did the AI make this prediction?")



    st.write(

        "The prototype performs a local sensitivity analysis. Each input "

        "is temporarily changed to a baseline value while the other "

        "inputs remain unchanged. The resulting probability change shows "

        "which inputs most affected this model prediction. This is an "

        "explanation of model behavior, not medical causation."

    )



    baseline_values = {

        "age": 30,

        "fever": 0,

        "cough": 0,

        "breathing_difficulty": 0,

        "chest_discomfort": 0,

        "vomiting": 0,

        "diarrhea": 0,

        "dizziness": 0,

        "fainting": 0,

        "severe_weakness": 0,

        "symptom_duration": 0,

        "diabetes": 0,

        "hypertension": 0,

        "heart_disease": 0,

        "asthma": 0,


    }



    current_probabilities = model.predict_proba(input_data)[0]

    prediction_index = classes.index(prediction)

    current_probability = current_probabilities[prediction_index]



    local_impacts = []



    for feature in feature_names:



        modified_data = input_data.copy()

        modified_data[feature] = baseline_values[feature]



        modified_probabilities = model.predict_proba(modified_data)[0]

        modified_probability = modified_probabilities[prediction_index]



        impact = current_probability - modified_probability



        local_impacts.append({

            "feature": display_names[feature],

            "impact": impact,

            "current_value": input_data.iloc[0][feature]

        })



    local_impacts.sort(

        key=lambda x: abs(x["impact"]),

        reverse=True

    )



    top_impacts = local_impacts[:5]



    for item in top_impacts:



        feature = item["feature"]

        impact = item["impact"]



        if impact > 0.001:



            st.write(

                f"🔴 **{feature}** — increasing this feature from "

                f"the baseline increased the predicted probability "

                f"of the selected risk class by **{impact * 100:.2f} "

                f"percentage points**."

            )



        elif impact < -0.001:



            st.write(

                f"🟢 **{feature}** — the current value decreased the "

                f"predicted probability of the selected risk class by "

                f"**{abs(impact) * 100:.2f} percentage points** "

                f"compared with the baseline."

            )



        else:



            st.write(

                f"⚪ **{feature}** — had a relatively small effect "

                f"on this specific prediction."

            )



    st.caption(

        "Note: This is a local model explanation for the current "

        "patient input. It does not establish medical causation."

    )





    # ========================================================

    # PATIENT SUMMARY

    # ========================================================



    st.divider()



    st.header("📋 Patient Summary")



    summary_col1, summary_col2, summary_col3 = st.columns(3)



    with summary_col1:

        st.metric("Age", age)



    with summary_col2:

        st.metric(

            "Symptom Duration",

            f"{symptom_duration} days"

        )



    with summary_col3:

        st.metric(

            "Severity Score",

            f"{severity_score}/10"

        )





    # ========================================================

    # GENERATE PATIENT REPORT

    # ========================================================

    st.divider()
    st.header("📄 Generate Patient Report")
    st.write(
        "Create a simple text report containing the patient inputs, "
        "model output, probabilities, and important symptom alerts. "
        "This report is for the hackathon prototype and is not a medical diagnosis."
    )

    symptom_report = [
        ("Fever", fever),
        ("Cough", cough),
        ("Breathing Difficulty", breathing_difficulty),
        ("Chest Discomfort", chest_discomfort),
        ("Vomiting", vomiting),
        ("Diarrhea", diarrhea),
        ("Dizziness", dizziness),
        ("Fainting", fainting),
        ("Severe Weakness", severe_weakness),
    ]

    history_report = [
        ("Diabetes", diabetes),
        ("Hypertension", hypertension),
        ("Heart Disease", heart_disease),
        ("Asthma", asthma),
    ]

    alert_lines = []
    if breathing_difficulty == "Yes":
        alert_lines.append("Breathing Difficulty")
    if chest_discomfort == "Yes":
        alert_lines.append("Chest Discomfort")
    if fainting == "Yes":
        alert_lines.append("Fainting")

    report_lines = [
        "RURALHEALTHAI - PATIENT RISK ASSESSMENT REPORT",
        "=" * 55,
        "",
        "IMPORTANT NOTICE",
        "This is a hackathon machine-learning prototype report.",
        "It does not provide a medical diagnosis and should not replace",
        "evaluation by a qualified healthcare professional.",
        "",
        "PATIENT INFORMATION",
        "-" * 25,
        f"Age: {age}",
        f"Symptom Duration: {symptom_duration} days",
        f"User-entered Severity Score: {severity_score}/10",
        "",
        "SYMPTOMS",
        "-" * 25,
    ]

    for name, value in symptom_report:
        report_lines.append(f"{name}: {value}")

    report_lines += ["", "MEDICAL HISTORY", "-" * 25]
    for name, value in history_report:
        report_lines.append(f"{name}: {value}")

    report_lines += [
        "",
        "AI MODEL RESULT",
        "-" * 25,
        f"Predicted Risk: {risk_name}",
        f"Low Risk Probability: {probability_map.get(0, 0) * 100:.1f}%",
        f"Medium Risk Probability: {probability_map.get(1, 0) * 100:.1f}%",
        f"High Risk Probability: {probability_map.get(2, 0) * 100:.1f}%",
        "",
        "IMPORTANT SYMPTOM ALERT",
        "-" * 25,
    ]

    if alert_lines:
        report_lines.append("Symptoms requiring prompt medical attention may include:")
        report_lines.extend(f"- {item}" for item in alert_lines)
        report_lines.append("Do not rely on the AI risk score to decide whether care is needed.")
    else:
        report_lines.append("No prototype emergency symptom trigger was selected.")

    report_lines += ["", "REPORT END", "RuralHealthAI Hackathon Prototype"]
    patient_report = "\n".join(report_lines)

    st.download_button(
        label="📥 Download Patient Report",
        data=patient_report,
        file_name="ruralhealthai_patient_report.txt",
        mime="text/plain",
        use_container_width=True,
    )


    # SAFETY MESSAGE

    # ========================================================



    st.divider()



    st.info(

        "⚠️ **Hackathon Prototype:** RuralHealthAI is a "

        "machine-learning decision-support prototype. It does "

        "not provide a medical diagnosis and should not replace "

        "evaluation by a qualified healthcare professional."

    )
