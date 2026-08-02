import numpy as np
from sklearn.neural_network import MLPClassifier
import streamlit as st

# --- 1. ENTRENAMIENTO DE LA RED NEURONAL (Del 1 al 50) ---
@st.cache_resource
def entrenar_modelo():
    X = np.array([[int(x) for x in list(np.binary_repr(n, width=6))] for n in range(51)])
    y = np.array([n % 2 for n in range(51)])
    
    clf = MLPClassifier(hidden_layer_sizes=(8,), activation='relu', max_iter=3000, random_state=42)
    clf.fit(X, y)
    return clf

modelo = entrenar_modelo()

def a_binario(n):
    return [int(x) for x in list(np.binary_repr(n, width=6))]

# --- 2. INTERFAZ GRÁFICA CON STREAMLIT ---
st.title("🤖 Clasificador de Números Pares e Impares con RNA")
st.write("Aplicación interactiva desarrollada con Redes Neuronales Artificiales en la nube por Rubén Alarcón Coria.")

st.info("Ingresa o selecciona un número entero del **1 al 50** para que la Red Neuronal prediga si es Par o Impar.")

# Selector numérico interactivo
numero_usuario = st.slider("Selecciona un número:", min_value=1, max_value=50, value=9)

if st.button("🔍 Probar con la Red Neuronal"):
    binario_lista = a_binario(numero_usuario)
    entrada = np.array([binario_lista])
    prediccion = modelo.predict(entrada)[0]
    
    resultado_texto = "IMPAR" if prediccion == 1 else "PAR"
    
    if prediccion == 1:
        st.warning(f"El número **{numero_usuario}** es **{resultado_texto}**.")
    else:
        st.success(f"El número **{numero_usuario}** es **{resultado_texto}**.")
        
    # --- DESGLOSE VISUAL DEL VECTOR BINARIO Y SUS PESOS ---
    st.write("### 📊 Análisis del Vector Binario (6 bits)")
    
    # Creamos columnas visuales para mostrar la cabecera de potencias y los bits abajo
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    pesos = [32, 16, 8, 4, 2, 1]
    
    with col1:
        st.metric(label="Bit (32)", value=binario_lista[0])
    with col2:
        st.metric(label="Bit (16)", value=binario_lista[1])
    with col3:
        st.metric(label="Bit (8)", value=binario_lista[2])
    with col4:
        st.metric(label="Bit (4)", value=binario_lista[3])
    with col5:
        st.metric(label="Bit (2)", value=binario_lista[4])
    with col6:
        st.metric(label="Bit (1)", value=binario_lista[5])
        
    st.caption(f"Vector completo ingresado a la RNA: `{binario_lista}`")
