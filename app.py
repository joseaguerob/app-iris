import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2
from datetime import datetime

USER = "postgres.mxpomiiojlifkraxuggs"
PASSWORD = "Messivuelvealbarcelonaporfa20"
HOST = "aws-1-us-east-2.pooler.supabase.com"
PORT = "6543"
DBNAME = "postgres"

st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")

@st.cache_resource
def load_models():
    try:
        model = joblib.load('components/iris_model.pkl')
        scaler = joblib.load('components/iris_scaler.pkl')
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo en la carpeta 'components/'")
        return None, None, None

st.title("🌸 Predictor de Especies de Iris")

model, scaler, model_info = load_models()

if model is not None:
    st.header("Ingresa las características de la flor:")
    
    longitud_sepalo = st.number_input("Longitud del Sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    ancho_sepalo = st.number_input("Ancho del Sépalo (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    longitud_petalo = st.number_input("Longitud del Pétalo (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    ancho_petalo = st.number_input("Ancho del Pétalo (cm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    if st.button("Predecir y Guardar"):
        features = np.array([[longitud_sepalo, ancho_sepalo, longitud_petalo, ancho_petalo]])
        
        features_scaled = scaler.transform(features)
        
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        target_names = model_info['target_names']
        predicted_species = target_names[prediction]
        
        st.success(f"Especie predicha: **{predicted_species}**")
        st.write(f"Confianza: **{max(probabilities):.1%}**")

        connection = None
        cursor = None
        try:
            connection = psycopg2.connect(
                user=USER,
                password=PASSWORD,
                host=HOST,
                port=PORT,
                dbname=DBNAME
            )
            
            cursor = connection.cursor()
            
            sql_query = """
            INSERT INTO "Table iris" ("longitud del petalo", "longitud del sepalo", "ancho del petalo", "ancho del sepalo", prediction)
            VALUES (%s, %s, %s, %s, %s)
            """
            
            values = (longitud_petalo, longitud_sepalo, ancho_petalo, ancho_petalo, predicted_species)
            
            cursor.execute(sql_query, values)
            
            connection.commit()
            
            st.success("¡Datos guardados en la tabla 'Table iris' de Supabase!")
            
            with st.expander("Mostrar información de depuración"):
                st.write("Consulta SQL ejecutada:")
                st.code(sql_query.strip())
                st.write("Valores insertados:")
                st.write(values)
                st.write("Hora de registro:")
                st.write(datetime.now())
            
        except Exception as e:
            st.error(f"Error al guardar en la base de datos: {e}")
            st.error("Revisa la consola para más detalles.")
            
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()
        
        st.write("Probabilidades:")
        for species, prob in zip(target_names, probabilities):
            st.write(f"- {species}: {prob:.1%}")

