import numpy as np
from sklearn.neural_network import MLPClassifier
import streamlit as st

# --- 1. ENTRENAMIENTO DE LA RED NEURONAL (Del 1 al 50) ---
# Usamos una representación de 6 bits para cubrir números hasta el 63 (suficiente para el 50)
@st.cache_resource
def entrenar_modelo():
    # Creamos un dataset del 0 al 50 en formato binario de 6 bits
    X = np.array([[int(x) for x in list(np.binary_repr(n, width=6))] for n in range(51)])
    # Etiquetas: 0 si es par, 1 si es impar
    y = np.array([n % 2 for n in range(51)])
    
    # Red Neuronal (Perceptrón Multicapa)
    clf = MLPClassifier(hidden_layer_sizes=(8,), activation='relu', max_iter=3000, random_state=42)
    clf.fit(X, y)
    return clf

modelo = entrenar_modelo()

# Función para convertir un número entero a su arreglo binario de 6 bits
def a_binario(n):
    return [int(x) for x in list(np.binary_repr(n, width=6))]

# --- 2. INTERFAZ GRÁFICA CON STREAMLIT ---
st.title("🤖 Clasificador de Números Pares e Impares con RNA")
st.write("Aplicación interactiva desarrollada con Redes Neuronales Artificiales en la nube.")

st.info("Ingresa o selecciona un número entero del **1 al 50** para que la Red Neuronal prediga si es Par o Impar.")

# Selector numérico interactivo (corregido min_value)
numero_usuario = st.slider("Selecciona un número:", min_value=1, max_value=50, value=15)

if st.button("🔍 Probar con la Red Neuronal"):
    # Convertimos el número del usuario a binario y hacemos la predicción
    entrada = np.array([a_binario(numero_usuario)])
    prediccion = modelo.predict(entrada)[0]
    
    resultado_texto = "IMPAR" if prediccion == 1 else "PAR"
    
    # Mostramos el resultado de forma visual en la pantalla
    if prediccion == 1:
        st.warning(f"El número **{numero_usuario}** es **{resultado_texto}**.")
    else:
        st.success(f"El número **{numero_usuario}** es **{resultado_texto}**.")
