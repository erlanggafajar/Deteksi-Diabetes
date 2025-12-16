import pickle
import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Aplikasi Prediksi Diabetes",
    layout="wide"    
)

# load model + scaler
model_data = pickle.load(open("diabetes_model_x.sav", "rb"))
diabetes_model = model_data["model"]
scaler = model_data["scaler"]

st.title("Data Mining Prediksi Diabetes")

col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.number_input("Pregnancies", min_value=0, step=1)

with col2:
    Glucose = st.number_input("Glucose", min_value=0)

with col1:
    BloodPressure = st.number_input("Blood Pressure", min_value=0)

with col2:
    SkinThickness = st.number_input("Skin Thickness", min_value=0)

with col1:
    Insulin = st.number_input("Insulin", min_value=0)

with col2:
    BMI = st.number_input("BMI", min_value=0.0, format="%.2f")

with col1:
    DiabetesPedigreeFunction = st.number_input(
        "Diabetes Pedigree Function", min_value=0.0, format="%.3f"
    )

with col2:
    Age = st.number_input("Age", min_value=0, step=1)

diab_diagnosis = ""

if st.button("Test Prediksi Diabetes"):
    input_data = np.array(
        [
            Pregnancies,
            Glucose,
            BloodPressure,
            SkinThickness,
            Insulin,
            BMI,
            DiabetesPedigreeFunction,
            Age,
        ]
    ).reshape(1, -1)

    # WAJIB distandarisasi
    input_data_std = scaler.transform(input_data)

    prediction = diabetes_model.predict(input_data_std)

    if prediction[0] == 1:
        diab_diagnosis = "Pasien terkena Diabetes"
    else:
        diab_diagnosis = "Pasien tidak terkena Diabetes"

st.success(diab_diagnosis)
