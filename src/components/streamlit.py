import pandas as pd
import pickle
import streamlit as st
from src.exception import CustomException
import sys
from collections import Counter

# Load model, scaler, and label encoder
with open("artifacts/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("artifacts/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("artifacts/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Encode categorical inputs before prediction
def encode_inputs(features):
    encoded_features = features.copy()

    for feature, encoder in label_encoders.items():
        if feature in features:
            try:
                encoded_features[feature] = encoder.transform([features[feature]])[0]
            except ValueError:
                st.warning(f"Unseen label '{features[feature]}' for '{feature}'. Using mode imputation.")
                try:
                    mode = Counter(encoder.classes_).most_common(1)[0][0]
                    encoded_features[feature] = encoder.transform([mode])[0]
                except ValueError:
                    st.warning(f"No classes found for the encoder of feature '{feature}'. Using 0 as default.")
                    encoded_features[feature] = 0
            except Exception as e:
                raise CustomException(f"Encoding error for '{feature}': {e}", sys)
    return encoded_features

# Predict mental health condition with encoding
def predict_condition(features):
    features_encoded = encode_inputs(features)
    features_df = pd.DataFrame([features_encoded])

    if features_df.empty:
        st.error("Please fill in at least one feature for prediction.")
        st.stop()

    features_scaled = scaler.transform(features_df)
    prediction = model.predict(features_scaled)[0]
    return prediction

# Streamlit app with encoding
def main():
    st.title("Mental Health Prediction System")
    st.write("Enter the following details to predict mental health condition:")

    features = {}
    try:
        age = int(st.text_input("Enter Age", value="18"))
        features["Age"] = age
    except ValueError:
        st.error("Invalid age input. Please enter a number.")
        st.stop()
        
    features["Gender"] = st.selectbox("Gender", ["male", "female", "trans", "other"])
    features["self_employed"] = st.selectbox("Self Employed", ["Yes", "No"])
    features["family_history"] = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
    features["work_interfere"] = st.selectbox("Work Interference", ["Never", "Rarely", "Sometimes", "Often"])
    features["no_employees"] = st.selectbox("Number of Employees", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
    features["remote_work"] = st.selectbox("Remote Work", ["Yes", "No"])
    features["tech_company"] = st.selectbox("Tech Company", ["Yes", "No"])
    features["benefits"] = st.selectbox("Benefits", ["Yes", "No", "Don't Know"])
    features["care_options"] = st.selectbox("Care Options", ["Yes", "No", "Not Sure"])
    features["wellness_program"] = st.selectbox("Wellness Program", ["Yes", "No", "Don't Know"])
    features["seek_help"] = st.selectbox("Seek Help", ["Yes", "No", "Don't Know"])
    features["anonymity"] = st.selectbox("Anonymity", ["Yes", "No"])
    features["mental_health_consequence"] = st.selectbox("Mental Health Consequence", ["Yes", "No", "Maybe"])
    features["phys_health_consequence"] = st.selectbox("Physical Health Consequence", ["Yes", "No", "Maybe"])
    features["coworkers"] = st.selectbox("Discuss with Coworkers", ["Yes", "No", "Some of Them"])
    features["supervisor"] = st.selectbox("Discuss with Supervisor", ["Yes", "No", "Some of Them"])
    features["mental_health_interview"] = st.selectbox("Mental Health Interview", ["Yes", "No", "Maybe"])
    features["phys_health_interview"] = st.selectbox("Physical Health Interview", ["Yes", "No", "Maybe"])
    features["mental_vs_physical"] = st.selectbox("Mental vs Physical", ["Yes", "No", "Don't Know"])
    features["obs_consequence"] = st.selectbox("Observed Consequences", ["Yes", "No"])

    if st.button("Predict"):
        try:
            prediction = predict_condition(features)
            condition = "Needs Treatment" if prediction == 1 else "No Treatment Needed"
            st.success(f"Predicted Mental Health Condition: {condition}")
        except Exception as e:
            st.exception(e)  # Display the exception details in Streamlit


if __name__ == "__main__":
    main()