# -*- coding: utf-8 -*-
"""
Created on Fri May 16 09:23:44 2025

@author: Sarthak
"""

import numpy as np
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open('S:/CODING/Heart Disease Detection/trained_model.sav', 'rb'))

# Prediction function
def heart_detetction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The Person does NOT have Heart Disease'
    else:
        return 'The Person HAS Heart Disease'

# Main function
def main():
    st.title('Heart Disease Detection Web App')

    # User input fields
    Age = st.text_input('Age of the Person')
    Sex = st.text_input('Gender (0 = Female, 1 = Male)')
    ConstrictivePericarditis = st.text_input('Chest Pain Type (CP Count)')
    BloodPressure = st.text_input('Resting Blood Pressure')
    Chorlestroll = st.text_input('Cholesterol Level')
    FastingBloodSugar = st.text_input('Fasting Blood Sugar (>120 mg/dl, 1 = True, 0 = False)')
    Electrocardiographic = st.text_input('Resting ECG Results')
    HeartRate = st.text_input('Maximum Heart Rate Achieved')
    ChestPain = st.text_input('Exercise Induced Angina (1 = Yes, 0 = No)')
    OldPeak = st.text_input('ST Depression Induced by Exercise')
    SlopeHeartrate = st.text_input('Slope of the Peak Exercise ST Segment')
    CaDisease = st.text_input('Number of Major Vessels Colored by Fluoroscopy (0–3)')
    Thalassemia = st.text_input('Thalassemia (1 = Normal; 2 = Fixed Defect; 3 = Reversible Defect)')

    # Prediction output
    diagnosis = ''

    if st.button('Heart Disease Test Result'):
        try:
            input_data = [
                float(Age),
                float(Sex),
                float(ConstrictivePericarditis),
                float(BloodPressure),
                float(Chorlestroll),
                float(FastingBloodSugar),
                float(Electrocardiographic),
                float(HeartRate),
                float(ChestPain),
                float(OldPeak),
                float(SlopeHeartrate),
                float(CaDisease),
                float(Thalassemia)
            ]
            diagnosis = heart_detetction(input_data)
        except ValueError:
            diagnosis = "⚠️ Please enter valid **numeric** values for all fields."

    st.success(diagnosis)

if __name__ == '__main__':
    main()
