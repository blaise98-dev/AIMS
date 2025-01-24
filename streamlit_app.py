import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
import joblib
import tensorflow as tf
warnings.filterwarnings('ignore')

# Initialize an empty list to store loaded models
models = []

# Load the LSTM model
try:
    lstm_model = tf.keras.models.load_model('/home/aimssn-it/Desktop/ResearchThesis/Codes/models/lstm_multiclass_model.h5')
    models.append(('LSTM', lstm_model))
    print("Loaded LSTM model")
except Exception as e:
    print(f"Error loading LSTM model: {e}")

# Load MLP model
try:
    mlp_model = tf.keras.models.load_model('/home/aimssn-it/Desktop/ResearchThesis/Codes/models/mlp_multiclass_model.h5')
    models.append(('MLP', mlp_model))
    print("Loaded MLP model")
except Exception as e:
    print(f"Error loading MLP model: {e}")

# Load the severity map
try:
    severity_map_path = "/home/aimssn-it/Desktop/ResearchThesis/Codes/models/file_severity_map.pkl"
    symptom_severity_map = joblib.load(severity_map_path)
    print("Loaded severity map")
except Exception as e:
    print(f"Error loading severity map: {e}")

# Load the label encoder
try:
    label_encoder_path = '/home/aimssn-it/Desktop/ResearchThesis/Codes/models/le_new.pkl'
    le = joblib.load(label_encoder_path)
    print("Loaded label encoder")
except Exception as e:
    print(f"Error loading label encoder: {e}")

# Load the training data (X_train)
try:
    X_train_path = '/home/aimssn-it/Desktop/ResearchThesis/Codes/models/file_X_train.pkl'
    X_train = joblib.load(X_train_path)
    print("Loaded X_train")
except Exception as e:
    print(f"Error loading X_train: {e}")

# Load the dataset to train the label encoder
data = pd.read_csv('/home/aimssn-it/Desktop/ResearchThesis/Codes/dataset/health_dataset_combined.csv')
labels = data['Disease']

# Encode the labels as integers
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Retrieve the disease names from the label encoder
disease_names = label_encoder.inverse_transform(np.arange(len(label_encoder.classes_)))

# Load precautionary measures
try:
    precaution_dict = joblib.load("/home/aimssn-it/Desktop/ResearchThesis/Codes/models/file_symptom_precaution_dict.pkl")
    print("Loaded precautionary measures")
except Exception as e:
    print(f"Error loading precautionary measures: {e}")

def preprocess_input(symptoms_list):
    # Preprocess the input symptoms
    symptoms_list = [symptom.lower().replace('(', '').replace(')', '').replace(' ', '_').replace(',', '_') for symptom in symptoms_list]
    return symptoms_list

def prepare_input_for_lstm(symptoms_list, feature_names):
    """
    Convert symptoms into the input format required for the LSTM model.
    Reshapes the input to match the model's expected shape.
    """
    feature_names = list(feature_names)  # Convert to list
    symptom_vector = np.zeros(len(feature_names))
    for symptom in symptoms_list:
        if symptom in feature_names:
            symptom_vector[feature_names.index(symptom)] = 1
    # Adjust the shape to match the model's expected input shape
    return np.array([symptom_vector]).reshape(1, 1, len(feature_names))  # Shape (1, 1, feature_size)

def prepare_input_for_mlp(symptoms_list, feature_names):
    """
    Convert symptoms into the input format required for the MLP model.
    """
    feature_names = list(feature_names)  # Convert to list
    symptom_vector = np.zeros(len(feature_names))
    for symptom in symptoms_list:
        if symptom in feature_names:
            symptom_vector[feature_names.index(symptom)] = 1
    return np.array([symptom_vector])  # Shape (1, feature_size)


def predict_disease(symptoms_list, model, model_name, top_k=3):
    feature_names = X_train.columns

    # Preprocess the input symptoms
    symptoms_list = preprocess_input(symptoms_list)

    if model_name == 'LSTM':
        input_data = prepare_input_for_lstm(symptoms_list, feature_names)
    else:
        input_data = prepare_input_for_mlp(symptoms_list, feature_names)

    if isinstance(model, tf.keras.Model):
        predicted_probs = model.predict(input_data)[0]  # Predict and get the first sample's output
    else:
        # Predict probabilities
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        predicted_probs = model.run([output_name], {input_name: input_data})[0][0]

    # Get disease names and probabilities
    disease_probs = list(zip(le.classes_, predicted_probs))

    # Sort by probability (descending)
    disease_probs.sort(key=lambda x: x[1], reverse=True)
    disease_probs = disease_probs[:top_k]

    # Calculate Average Severity for each disease using symptom_severity_map
    disease_severities = []
    for disease, prob in disease_probs:
        avg_severity = np.mean([symptom_severity_map.get(symptom, 0) for symptom in symptoms_list])
        disease_severities.append((disease, prob, avg_severity))

    # Adjust Probabilities to sum to 1, weighted by severity
    total_prob = sum(prob for _, prob, _ in disease_severities)
    adjusted_probs = [
        (disease, (prob / total_prob) * (1 + (avg_severity - 1) / 6), avg_severity)
        for disease, prob, avg_severity in disease_severities
    ]

    # Re-normalize probabilities to sum to 1
    total_adjusted_prob = sum(prob for _, prob, _ in adjusted_probs)
    adjusted_probs = [
        (disease, prob / total_adjusted_prob, avg_severity)
        for disease, prob, avg_severity in adjusted_probs
    ]

    # Sort by adjusted probability (descending)
    adjusted_probs.sort(key=lambda x: x[1], reverse=True)

    return adjusted_probs

def output_predicted_diseases(symptoms_list, model, model_name, top_k=3):
    top_k_diseases = predict_disease(symptoms_list, model, model_name, top_k)

    # Display each disease and probability as a tuple in a list format
    st.write("### The results indicate that you may be suffering from:")
    for disease, prob, severity in top_k_diseases:
        st.markdown(f"<span style='font-size:26px;'>{prob * 100:.0f}% shows <span style='color:brown;'>{disease}</span></span>", unsafe_allow_html=True)

    return top_k_diseases

# Streamlit app
st.set_page_config(page_title="SmartHealth: Disease Prediction and Precaution Tracker", page_icon=":microbe:", layout="wide")
st.title("SmartHealth: Disease Prediction and Precaution Tracker")

# Sidebar for symptoms and model selection
with st.sidebar:
    st.write("### Select Symptoms")
    symptom_options = list(symptom_severity_map.keys())
    symptoms_list = st.multiselect("", symptom_options, placeholder="Select symptoms")

    if models:
        st.write("### Select a Model")
        selected_model_name = st.selectbox("", [name for name, _ in models], placeholder="Select a Model")
        selected_model = next(model for name, model in models if name == selected_model_name)
    else:
        st.error("No models are loaded. Please check the model files.")

# Main page layout
# First row: Predicted Diseases
if st.button("Predict"):
    if not symptoms_list:
        st.warning("Please select at least one symptom.")
    elif not models:
        st.error("No models available for prediction.")
    else:
        try:
            top_k_diseases = output_predicted_diseases(symptoms_list, selected_model, selected_model_name)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Second row: Precautionary Measures
if 'top_k_diseases' in locals():
    st.write("### The following Precautions are recommended:")
    cols = st.columns(3, gap="small")
    for i, (disease, prob, severity) in enumerate(top_k_diseases):
        with cols[i % 3]:
            st.markdown(f"<span style='color:brown; font-weight:bold; font-size:22px;'>{disease}</span>", unsafe_allow_html=True)
            if disease in precaution_dict:
                for precaution in precaution_dict[disease]:
                    st.markdown(f"<span style='font-size:26px;'>- {precaution}</span>", unsafe_allow_html=True)
            else:
                st.write("No precautionary measures available.")