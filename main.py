import numpy as np
from sklearn.neural_network import MLPClassifier

# 1. Preparar el dataset de entrenamiento
# Entradas: Números del 0 al 9 representados como matrices de características binarias (bits)
# O simplemente usando directamente el valor entero si la red multicapa lo procesa.
# Vamos a usar una representación binaria de 4 bits para los números del 0 al 15:
# Ejemplo: 0 = [0,0,0,0], 1 = [0,0,0,1], 2 = [0,0,0,10], etc., o directo con valores numéricos.
# Usemos una forma más directa con números enteros y una matriz de características simples:
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]])

# Salidas esperadas: 0 para Par, 1 para Impar
y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

# 2. Crear el modelo de Red Neuronal Artificial (Perceptrón Multicapa)
# Una capa oculta con 4 neuronas y función de activación ReLU / logistic
clf = MLPClassifier(hidden_layer_sizes=(4,), activation='relu', max_iter=1000, random_state=42)

# 3. Entrenar la red neuronal
print("Entrenando la Red Neuronal Artificial...")
clf.fit(X, y)
print("¡Entrenamiento completado con éxito!\n")

# 4. Probar el modelo con nuevos números
numeros_a_probar = np.array([[13], [16], [21], [28]])
predicciones = clf.predict(numeros_a_probar)

print("--- PRUEBAS DE PREDICCIÓN ---")
for num, pred in zip(numeros_a_probar, predicciones):
    resultado = "Impar" if pred == 1 else "Par"
    print(f"Número: {num[0]} -> Predicción de la RNA: {resultado} ({pred})")
