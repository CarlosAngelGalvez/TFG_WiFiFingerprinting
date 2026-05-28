# 03_escenario_base.py
# Este script entrena los tres modelos (KNN, Random Forest y MLP)
# con los datos preprocesados y los evalúa sobre el conjunto de validación
# sin ninguna perturbación. Los resultados de este escenario base sirven
# como referencia para cuantificar la degradación en los escenarios
# de ruido gaussiano y AP missing (apartado 3.2.1 de la memoria).

import numpy as np
import pickle
import time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# Cargo los datos preprocesados que generó el script 02.
X_train = np.load('results/processed/X_train.npy')
X_val = np.load('results/processed/X_val.npy')
y_train = np.load('results/processed/y_train.npy')
y_val = np.load('results/processed/y_val.npy')
y_train_scaled = np.load('results/processed/y_train_scaled.npy')

# Cargo el scaler de las variables objetivo, que necesito para
# deshacer la transformación de las predicciones del MLP.
with open('results/processed/scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

# Función que implementa las métricas definidas en el apartado 3.5.
# Primero calcula la distancia euclídea entre posición real y estimada
# para cada muestra (ecuación 3.2), y a partir de ahí obtiene
# el MAE (ecuación 3.3) y el RMSE (ecuación 3.4).
def calcular_metricas(y_real, y_pred):
    distancias = np.sqrt(np.sum((y_real - y_pred) ** 2, axis=1))
    mae = np.mean(distancias)
    rmse = np.sqrt(np.mean(distancias ** 2))
    return distancias, mae, rmse

# Defino los tres modelos con los hiperparámetros de la Tabla 3.2.
# KNN: k=5, ponderación por distancia, distancia euclídea.
#   No tiene fase de entrenamiento como tal, almacena los datos
#   y busca los vecinos más cercanos en tiempo de predicción.
# Random Forest: 100 árboles, profundidad sin límite, semilla 42.
#   n_jobs=-1 usa todos los núcleos del procesador para acelerar.
# MLP: dos capas ocultas (128, 64), ReLU, Adam, 500 épocas máximo.
#   La parada temprana con paciencia 10 detiene el entrenamiento
#   si la pérdida no mejora en 10 épocas consecutivas,
#   lo que previene el sobreajuste.
modelos = {
    'KNN': KNeighborsRegressor(
        n_neighbors=5,
        weights='distance',
        metric='euclidean'
    ),
    'Random Forest': RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ),
    'MLP': MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42
    )
}

# Entreno y evalúo cada modelo. Mido por separado los tiempos
# de entrenamiento y validación para analizar el coste
# computacional de cada fase de forma independiente.
print("=" * 60)
print("ESCENARIO BASE - Resultados")
print("=" * 60)
print(f"{'Modelo':<16} {'MAE (m)':>10} {'RMSE (m)':>10} {'Train (s)':>12} {'Val (s)':>10}")
print("-" * 60)

resultados_base = {}

for nombre, modelo in modelos.items():
    inicio = time.time()

# El MLP se entrena con las coordenadas escaladas (StandardScaler)
# porque las magnitudes UTM originales impiden que Adam converja.
# Después de predecir, deshago el escalado con inverse_transform
# para obtener las coordenadas en metros y calcular las métricas.
# KNN y Random Forest trabajan directamente con las coordenadas originales.
#
# Se miden por separado los tiempos de entrenamiento y validación
# para analizar el coste computacional de cada fase. Esta distinción
# es relevante porque KNN no tiene entrenamiento real (solo almacena
# los datos), mientras que su validación es más costosa al requerir
# el cálculo de distancias contra todas las muestras de entrenamiento.

    if nombre == 'MLP':
        # Entrenamiento: ajuste de pesos de la red
        inicio_train = time.time()
        modelo.fit(X_train, y_train_scaled)
        tiempo_train = time.time() - inicio_train

        # Validación: predicción sobre datos no vistos + desescalado
        inicio_val = time.time()
        y_pred_scaled = modelo.predict(X_val)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        tiempo_val = time.time() - inicio_val
    else:
        # Entrenamiento: ajuste del modelo
        inicio_train = time.time()
        modelo.fit(X_train, y_train)
        tiempo_train = time.time() - inicio_train

        # Validación: predicción sobre datos no vistos
        inicio_val = time.time()
        y_pred = modelo.predict(X_val)
        tiempo_val = time.time() - inicio_val

    distancias, mae, rmse = calcular_metricas(y_val, y_pred)
    resultados_base[nombre] = {
        'mae': mae,
        'rmse': rmse,
        'distancias': distancias,
        'tiempo_train': tiempo_train,
        'tiempo_val': tiempo_val
    }

    print(f"{nombre:<16} {mae:>10.2f} {rmse:>10.2f} {tiempo_train:>12.2f} {tiempo_val:>10.4f}")

# Guardo los resultados para usarlos en el script de gráficas (06).
np.save('results/processed/resultados_base.npy', resultados_base, allow_pickle=True)

print("\nResultados guardados en results/processed/resultados_base.npy")