# 05_ap_missing.py
# Este script evalúa los tres modelos bajo la perturbación de pérdida
# de puntos de acceso descrita en el apartado 3.2.3 de la memoria.
# Se simula la desaparición de APs eliminando columnas completas
# del conjunto de validación, lo que reproduce situaciones reales
# en entornos industriales donde un AP puede caer por fallo de hardware,
# reconfiguración de la red o cambios en la disposición de la planta.
# Se prueban tres niveles de degradación: p = 10%, 20% y 30%.

import numpy as np
import pickle
import pandas as pd
import time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# Cargo los datos de entrenamiento ya preprocesados.
X_train = np.load('results/processed/X_train.npy')
y_train = np.load('results/processed/y_train.npy')
y_val = np.load('results/processed/y_val.npy')
y_train_scaled = np.load('results/processed/y_train_scaled.npy')

# Cargo los scalers para aplicar el mismo preprocesado
# a los datos perturbados.
with open('results/processed/scaler.pkl', 'rb') as f:
    scaler_x = pickle.load(f)

with open('results/processed/scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

# Al igual que en el script de ruido, cargo los datos de validación
# originales (sin preprocesar) porque la perturbación se aplica
# sobre los valores RSSI crudos, antes de la normalización.
df_val = pd.read_csv('data/validationData.csv')
wap_cols = [col for col in df_val.columns if col.startswith('WAP')]
X_val_original = df_val[wap_cols].values.astype(float)

# Número total de WAPs, necesario para calcular cuántos eliminar
# en cada nivel de degradación.
n_waps = len(wap_cols)

# Misma función de métricas que en los scripts anteriores.
def calcular_metricas(y_real, y_pred):
    distancias = np.sqrt(np.sum((y_real - y_pred) ** 2, axis=1))
    mae = np.mean(distancias)
    rmse = np.sqrt(np.mean(distancias ** 2))
    return distancias, mae, rmse

# Entreno los modelos una sola vez, igual que en el script de ruido.
# La perturbación solo afecta a validación: el modelo ya está entrenado
# y se enfrenta a datos degradados en tiempo de predicción.
# Mido el tiempo de entrenamiento de cada modelo para comparar
# su coste computacional en la fase de ajuste.
print("Entrenando modelos...")

modelos_entrenados = {}
tiempos_train = {}
modelos_def = {
    'KNN': KNeighborsRegressor(
        n_neighbors=5, weights='distance', metric='euclidean'
    ),
    'Random Forest': RandomForestRegressor(
        n_estimators=100, max_depth=None, random_state=42, n_jobs=-1
    ),
    'MLP': MLPRegressor(
        hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
        max_iter=500, early_stopping=True, n_iter_no_change=10, random_state=42
    )
}

for nombre, modelo in modelos_def.items():
    inicio_train = time.time()
    if nombre == 'MLP':
        modelo.fit(X_train, y_train_scaled)
    else:
        modelo.fit(X_train, y_train)
    tiempos_train[nombre] = time.time() - inicio_train
    modelos_entrenados[nombre] = modelo

print("Modelos entrenados.\n")

# Evalúo cada modelo bajo los tres niveles de pérdida de APs.
# Para cada porcentaje, selecciono aleatoriamente qué APs desaparecen
# y sustituyo sus valores por el centinela 100 en TODAS las muestras
# de validación. La eliminación se hace por columnas completas,
# no por muestra individual, porque cuando un AP cae en una fábrica
# deja de estar disponible para todas las mediciones.
porcentajes = [10, 20, 30]
resultados_missing = {}

print("=" * 60)
print("AP MISSING - Resultados")
print("=" * 60)

for p in porcentajes:
    # Calculo cuántos APs hay que eliminar para este porcentaje.
    # Por ejemplo, 10% de 520 = 52 APs eliminados.
    n_eliminar = int(n_waps * p / 100)

    print(f"\n--- p = {p}% ({n_eliminar} APs eliminados de {n_waps}) ---")
    print(f"{'Modelo':<16} {'MAE (m)':>10} {'RMSE (m)':>10} {'Val (s)':>10}")
    print("-" * 48)

    resultados_missing[p] = {}

    # Selecciono aleatoriamente qué APs se eliminan.
    # La semilla 42 garantiza que siempre se eliminan los mismos APs,
    # lo que hace el experimento reproducible.
    np.random.seed(42)
    aps_eliminados = np.random.choice(n_waps, size=n_eliminar, replace=False)

    # Aplico la perturbación: pongo a 100 (centinela de ausencia)
    # todas las filas de las columnas correspondientes a los APs eliminados.
    X_val_missing = X_val_original.copy()
    X_val_missing[:, aps_eliminados] = 100

    # Aplico el mismo preprocesado que en el escenario base:
    # centinela 100 → -110 dBm, y después normalización Min-Max
    # con el scaler ajustado en entrenamiento.
    X_val_missing[X_val_missing == 100] = -110
    X_val_norm = scaler_x.transform(X_val_missing)

    for nombre, modelo in modelos_entrenados.items():
        # Mido el tiempo de validación para cada modelo y nivel de AP missing.
        inicio_val = time.time()
        if nombre == 'MLP':
            y_pred_scaled = modelo.predict(X_val_norm)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
        else:
            y_pred = modelo.predict(X_val_norm)
        tiempo_val = time.time() - inicio_val

        distancias, mae, rmse = calcular_metricas(y_val, y_pred)
        resultados_missing[p][nombre] = {
            'mae': mae, 'rmse': rmse, 'distancias': distancias,
            'tiempo_train': tiempos_train[nombre],
            'tiempo_val': tiempo_val
        }

        print(f"{nombre:<16} {mae:>10.2f} {rmse:>10.2f} {tiempo_val:>10.4f}")

# Guardo los resultados para las gráficas del script 06.
np.save('results/processed/resultados_missing.npy', resultados_missing, allow_pickle=True)
print("\nResultados guardados en results/processed/resultados_missing.npy")