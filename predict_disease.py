import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# Load the models and data
models = {}
for model_name in ['Random Forest', 'SVC']:
    with open(f'/home/aimssn-it/Desktop/ResearchThesis/Codes/models/best_model_{model_name}.pkl', 'rb') as file:
        models[model_name] = pickle.load(file)

with open("/home/aimssn-it/Desktop/ResearchThesis/Codes/models/severity_map.pkl", "rb") as file:
    symptom_severity_map = pickle.load(file)

with open('/home/aimssn-it/Desktop/ResearchThesis/Codes/models/le.pkl', 'rb') as f:
    le = pickle.load(f)

with open('/home/aimssn-it/Desktop/ResearchThesis/Codes/models/X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)

# Load the dataset to train the label encoder
data = pd.read_csv('/home/aimssn-it/Desktop/ResearchThesis/Codes/dataset/health_dataset_combined.csv')
labels = data['Disease']

# Encode the labels as integers
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Retrieve the disease names from the label encoder
disease_names = label_encoder.inverse_transform(np.arange(len(label_encoder.classes_)))

# Load precautionary measures
with open('/home/aimssn-it/Desktop/ResearchThesis/Codes/models/precaution_dict.pkl', 'rb') as f:
    precaution_dict = pickle.load(f)

def predict_disease(symptoms_list, model, top_k=3):
    feature_names = X_train.columns

    # 1. Create symptom DataFrame
    symptom_df = pd.DataFrame(0, index=[0], columns=feature_names)
    for symptom in symptoms_list:
        if symptom in feature_names:
            symptom_df.loc[0, symptom] = 1

    # 2. Predict probabilities
    if hasattr(model, "predict_proba"):
        predicted_probs = model.predict_proba(symptom_df)[0]
    elif hasattr(model, "decision_function"):
        decision_values = model.decision_function(symptom_df)[0]
        predicted_probs = np.exp(decision_values) / np.sum(np.exp(decision_values))
    else:
        raise ValueError("Model lacks predict_proba or decision_function.")

    # 3. Get disease names and probabilities
    disease_probs = list(zip(le.classes_, predicted_probs))

    # 4. Sort by probability (descending)
    disease_probs.sort(key=lambda x: x[1], reverse=True)

    return disease_probs[:top_k]

# Streamlit app
st.set_page_config(page_title="SmartHealth: Disease Prediction and Precaution Tracker", 
                   page_icon=":microbe:", layout="wide")
st.title("SmartHealth: Disease Prediction and Precaution Tracker")

# Sidebar for symptoms and model selection
with st.sidebar:
    st.subheader("Select Symptoms")
    symptom_options = list(symptom_severity_map.keys())
    symptoms_list = st.multiselect("Symptoms", symptom_options, placeholder="Select symptoms")

    if models:
        st.subheader("Select a Model")
        selected_model_name = st.selectbox("Model", list(models.keys()), placeholder="Select a model")
        selected_model = models[selected_model_name]
    else:
        st.error("No models are loaded. Please check the model files.")

# Main page layout
if st.button("Predict"):
    if not symptoms_list:
        st.warning("Please select at least one symptom.")
    elif not models:
        st.error("No models available for prediction.")
    else:
        try:
            # Predict top diseases
            top_k_diseases = predict_disease(symptoms_list, selected_model, top_k=3)

            # Display prediction results
            st.subheader(f"Predicted Diseases using {selected_model_name}")
            disease_names = [disease[0] for disease in top_k_diseases]
            disease_probs = [disease[1] for disease in top_k_diseases]

            fig = px.bar(
                x=disease_names,
                y=disease_probs,
                title="Predicted Diseases and Probabilities",
                labels={"x": "Diseases", "y": "Probability"},
            )
            st.plotly_chart(fig, use_container_width=True)

            # Display precautions
            st.subheader("Recommended Precautions")
            for disease, prob in top_k_diseases:
                st.markdown(
                    f"<span style='color:brown; font-weight:bold; font-size:22px;'>{disease}</span>", 
                    unsafe_allow_html=True
                )
                if disease in precaution_dict:
                    precautions = precaution_dict[disease]
                    for precaution in precautions:
                        st.markdown(f"- {precaution}")
                else:
                    st.write("No precautionary measures available.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
