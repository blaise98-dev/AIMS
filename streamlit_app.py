import streamlit as st
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder

# Load the models and data
def load_models():
    base_path = '/home/aimssn-it/Desktop/ResearchThesis/Codes/models/'
    
    def load_pickle(file_name):
        file_path = os.path.join(base_path, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            st.error(f"File not found: {file_path}")
            st.stop()

    symptoms = load_pickle('symptoms.pkl')
    rfc = load_pickle('KNeighborsClassifier.pkl')
    severity_map = load_pickle('severity_map.pkl')
    precaution_dict = load_pickle('precaution_dict.pkl')

    return symptoms, rfc, severity_map, precaution_dict

# Load the models and other necessary data
symptoms, rf, severity_map, precaution_dict = load_models()

# Load the dataset to train the label encoder
data = pd.read_csv('/home/aimssn-it/Desktop/ResearchThesis/Codes/dataset/dataset_modified.csv')
labels = data['Disease']

# Encode the labels as integers
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Function to predict disease
def predict_disease(symptoms_list, model):
    encoded_symptoms = [severity_map.get(symptom, 0) for symptom in symptoms_list]
    for i in range(len(encoded_symptoms), 17):
        encoded_symptoms.append(0)
    symptoms_array = np.array(encoded_symptoms).reshape(1, -1)
    disease_probs = model.predict_proba(symptoms_array)[0]
    top_disease_idx = np.argmax(disease_probs)
    top_disease = label_encoder.inverse_transform([top_disease_idx])[0]
    top_probability = round(disease_probs[top_disease_idx], 2)
    return top_disease, top_probability

# Streamlit app
st.title('Disease Prediction System')

# Multiselect for symptoms
selected_symptoms = st.multiselect('Select your symptoms', symptoms)

# Button to predict disease
if st.button('Predict Disease'):
    if selected_symptoms:
        top_disease, top_probability = predict_disease(selected_symptoms, rf)
        st.write(f"**Predicted Disease:** {top_disease}")
        st.write(f"**Probability:** {top_probability}")
        
        # Display the precautionary measures
        st.write("### Precautionary Measures:")
        if top_disease in precaution_dict:
            st.write(f"**{top_disease}:**")
            for precaution in precaution_dict[top_disease]:
                st.write(f"- {precaution}")
    else:
        st.write("Please select at least one symptom.")