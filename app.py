import streamlit as st
import joblib
import pickle
import numpy as np

import psycopg2


 USER = "postgres.mxpomiiojlifkraxuggs" #os.getenv("user")
 PASSWORD = "Messivuelvealbarcelonaporfa20"# os.getenv("password")
 HOST = "aws-1-us-east-2.pooler.supabase.com" #os.getenv("host")
 PORT = "6543" #os.getenv("port")
 DBNAME = "postgres" #os.getenv("dbname")


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
    
    sepal_length = st.number_input("Longitud del Sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width = st.number_input("Ancho del Sépalo (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    petal_length = st.number_input("Longitud del Pétalo (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    petal_width = st.number_input("Ancho del Pétalo (cm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    # Botón de predicción
    if st.button("Predecir y Guardar"):
        # Preparar datos
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
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
        
        # Mostrar todas las probabilidades
        st.write("Probabilidades:")
        for species, prob in zip(target_names, probabilities):
            st.write(f"- {species}: {prob:.1%}")

        # --- Guardar la predicción en Supabase ---
        try:
            # Conectar a la base de datos
            connection = psycopg2.connect(
                user=USER,
                password=PASSWORD,
                host=HOST,
                port=PORT,
                dbname=DBNAME
            )
            
            # Crear un cursor
            cursor = connection.cursor()
            
            # Definir la consulta SQL para insertar los datos
            sql_query = """
            INSERT INTO "Table Iris" (sepal_length, sepal_width, petal_length, petal_width, predicted_species)
            VALUES (%s, %s, %s, %s, %s);
            """
            
            # Definir los valores a insertar
            values = (sepal_length, sepal_width, petal_length, petal_width, predicted_species)
            
            # Ejecutar la consulta
            cursor.execute(sql_query, values)
            
            # Confirmar los cambios
            connection.commit()
            
            st.success("¡Datos guardados en la tabla 'Table Iris' de Supabase!")
        
        except Exception as e:
            st.error(f"Error al guardar en la base de datos: {e}")
        
        finally:
            # Cerrar la conexión y el cursor
            if 'cursor' in locals() and cursor:
                cursor.close()
            if 'connection' in locals() and connection:
                connection.close()
