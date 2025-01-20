from utils import db_connect
import streamlit as st
import pickle
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(current_dir, "random_forest_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

scaler_path = os.path.join(current_dir, "scaler.pkl")
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

label_encoders_path = os.path.join(current_dir, "label_encoders.pkl")
with open(label_encoders_path, "rb") as f:
    label_encoders = pickle.load(f)

def prepare_features(form_data, scaler, label_encoders):
    area = float(form_data['area'])
    bedrooms = int(form_data['bedrooms'])
    bathrooms = int(form_data['bathrooms'])
    stories = int(form_data['stories'])
    parking = int(form_data['parking'])
    mainroad = 1 if form_data['mainroad'] == 'yes' else 0
    guestroom = 1 if form_data['guestroom'] == 'yes' else 0
    basement = 1 if form_data['basement'] == 'yes' else 0
    hotwaterheating = 1 if form_data['hotwaterheating'] == 'yes' else 0
    airconditioning = 1 if form_data['airconditioning'] == 'yes' else 0
    prefarea = 1 if form_data['prefarea'] == 'yes' else 0
    furnishingstatus = label_encoders['furnishingstatus'].transform([form_data['furnishingstatus']])[0]
    price_category = label_encoders['price_category'].transform([form_data['price_category']])[0]

    features = np.array([[area, bedrooms, bathrooms, stories, parking, mainroad, guestroom, basement,
                          hotwaterheating, airconditioning, prefarea, furnishingstatus, price_category]])

    return scaler.transform(features)

st.title("Predicción de Precio de Vivienda")

with st.form("prediction_form"):
    st.header("Introduce los detalles de la vivienda:")
    
    area = st.number_input("Área (en pies cuadrados):", min_value=0.0, step=0.1)
    bedrooms = st.number_input("Número de dormitorios:", min_value=0, step=1)
    bathrooms = st.number_input("Número de baños:", min_value=0, step=1)
    stories = st.number_input("Número de pisos:", min_value=0, step=1)
    parking = st.number_input("Espacios de estacionamiento:", min_value=0, step=1)
    
    mainroad = st.selectbox("¿Tiene acceso a la calle principal?", ["yes", "no"])
    guestroom = st.selectbox("¿Tiene habitación para invitados?", ["yes", "no"])
    basement = st.selectbox("¿Tiene sótano?", ["yes", "no"])
    hotwaterheating = st.selectbox("¿Tiene calentador de agua?", ["yes", "no"])
    airconditioning = st.selectbox("¿Tiene aire acondicionado?", ["yes", "no"])
    prefarea = st.selectbox("¿Es un área preferencial?", ["yes", "no"])
    
    furnishingstatus = st.selectbox("Estado del mobiliario:", label_encoders['furnishingstatus'].classes_)
    price_category = st.selectbox("Categoría de precio:", label_encoders['price_category'].classes_)
    
    submitted = st.form_submit_button("Predecir")

if submitted:
    try:
        form_data = {
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'parking': parking,
            'mainroad': mainroad,
            'guestroom': guestroom,
            'basement': basement,
            'hotwaterheating': hotwaterheating,
            'airconditioning': airconditioning,
            'prefarea': prefarea,
            'furnishingstatus': furnishingstatus,
            'price_category': price_category,
        }
        
        features_scaled = prepare_features(form_data, scaler, label_encoders)
        
        predicted_price = model.predict(features_scaled)[0]
        
        st.success(f"El precio predicho de la vivienda es: {predicted_price}")
    
    except Exception as e:
        st.error(f"Ocurrió un error al procesar los datos: {e}")

engine = db_connect()
