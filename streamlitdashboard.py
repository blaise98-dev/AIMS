import streamlit as st
import pickle
import numpy as np
from predict_disease import predict_disease

# Load the models and data
with open('models/symptoms.plk', 'rb') as f:
    symptoms = pickle.load(f)

with open('models/RandomForestClassifier.plk', 'rb') as f:
    rfc = pickle.load(f)

with open('./models/precaution_dict.plk', 'rb') as f:
    precaution_dict = pickle.load(f)

# Title and description
st.title('Disease Diagnosis System')
st.write('Enter the symptoms that you have been experiencing lately and click on submit to get the prediction.')

# Input fields for symptoms
user_input = st.multiselect('Select your symptoms', symptoms)

# Prediction and display results
if st.button('Submit'):
    if user_input:
        prediction = predict_disease(user_input, rfc)
        st.write(f'You may have: **{prediction[0][0]}**')
        
        st.write('Here are some of the possible diseases that you may have:')
        for disease, prob in prediction:
            st.write(f'- {disease} ({prob*100:.2f}%)')
        
        st.write('You need to follow this immediately:')
        for precaution in precaution_dict[prediction[0][0]]:
            st.write(f'- {precaution}')
    else:
        st.write('Please select at least one symptom.')