import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2
from datetime import datetime

# Credenciales de la base de datos
USER = "postgres.mxpomiiojlifkraxuggs"
PASSWORD = "Messivuelvealbarcelonaporfa20"
HOST = "aws-1-us-east-2.pooler.supabase.com"
PORT = "6543"
DBNAME = "postgres"

# Configuración de la página
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")

# Función para cargar los modelos
@st.cache_resource
def load_models():
    """Carga los modelos de machine learning."""
    try:
        model = joblib.load('components/iris_model.pkl')
        scaler = joblib.load('components/iris_scaler.pkl')
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo en la carpeta 'components/'")
        return None, None, None

# Título
st.title("🌸 Predictor de Especies de Iris")

# Cargar modelos
model, scaler, model_info = load_models()

if model is not None:
    # Inputs
    st.header("Ingresa las características de la flor:")
    
    longitud_sepalo = st.number_input("Longitud del Sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    ancho_sepalo = st.number_input("Ancho del Sépalo (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    longitud_petalo = st.number_input("Longitud del Pétalo (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    ancho_petalo = st.number_input("Ancho del Pétalo (cm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    # Botón de predicción
    if st.button("Predecir y Guardar"):
        # Preparar datos
        features = np.array([[longitud_sepalo, ancho_sepalo, longitud_petalo, ancho_petalo]])
        
        # Estandarizar
        features_scaled = scaler.transform(features)
        
        # Predecir
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Mostrar resultado
        target_names = model_info['target_names']
        predicted_species = target_names[prediction]
        
        st.success(f"Especie predicha: **{predicted_species}**")
        st.write(f"Confianza: **{max(probabilities):.1%}**")

        # Guardar en la base de datos
        connection = None
        cursor = None
        try:
            # Conectar a la base de datos
            connection = psycopg2.connect(
                user=USER,
                password=PASSWORD,
                host=HOST,
                port=PORT,
                dbname=DBNAME
            )
            
            # Crear un cursor para ejecutar consultas
            cursor = connection.cursor()
            
            # Consulta SQL para insertar datos, incluyendo la fecha y hora
            # NOTA: Asegúrate de que tu tabla en Supabase tenga una columna llamada
            # 'prediction_timestamp' de tipo TIMESTAMP.
            sql_query = """
            INSERT INTO table_iris (longitud_petalo, longitud_sepalo, ancho_petalo, ancho_sepalo, prediction, prediction_timestamp)
            VALUES (%s, %s, %s, %s, %s, NOW())
            """
            
            # Los valores a insertar
            values = (longitud_petalo, longitud_sepalo, ancho_petalo, ancho_sepalo, predicted_species)
            
            # Ejecutar la consulta
            cursor.execute(sql_query, values)
            
            # Confirmar los cambios
            connection.commit()
            
            st.success("¡Datos guardados en la tabla 'table_iris' de Supabase!")
            
            # Bloque único para mostrar la consulta SQL
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
            # Asegurar que el cursor y la conexión se cierren
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()
        
        # Mostrar todas las probabilidades
        st.write("Probabilidades:")
        for species, prob in zip(target_names, probabilities):
            st.write(f"- {species}: {prob:.1%}")
