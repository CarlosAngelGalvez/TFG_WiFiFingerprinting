# 04_ruido_gaussiano.py
# Este script evalúa los tres modelos bajo la perturbación de ruido
# gaussiano descrita en el apartado 3.2.2 de la memoria.
# El ruido se aplica SOLO sobre el conjunto de validación, simulando
# las fluctuaciones de señal que ocurren en entornos reales por
# interferencias electromagnéticas o efectos de multitrayecto.
# Se prueban tres niveles de intensidad: σ = 2, 4 y 6 dBm.
# Los modelos se entrenan una sola vez (sin ruido) y se evalúan
# sobre los datos perturbados, porque en un despliegue real el modelo
# ya estaría entrenado y recibiría señales degradadas.

import numpy as np
import pickle
import time
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import csv

# Cargo los datos de entrenamiento ya preprocesados.
X_train = np.load('results/processed/X_train.npy')
y_train = np.load('results/processed/y_train.npy')
y_val = np.load('results/processed/y_val.npy')
y_train_scaled = np.load('results/processed/y_train_scaled.npy')

# Necesito los scalers para preprocesar los datos perturbados
# de la misma forma que los datos originales.
with open('results/processed/scaler.pkl', 'rb') as f:
    scaler_x = pickle.load(f)

with open('results/processed/scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

# Cargo los datos de validación ORIGINALES (antes de preprocesar)
# porque el ruido se tiene que aplicar sobre los valores RSSI reales,
# no sobre los normalizados. Después de añadir el ruido,
# aplico el mismo preprocesado (centinela → -110 + Min-Max).
df_val = pd.read_csv('data/validationData.csv')
wap_cols = [col for col in df_val.columns if col.startswith('WAP')]
X_val_original = df_val[wap_cols].values.astype(float)

# Misma función de métricas que en el escenario base.
def calcular_metricas(y_real, y_pred):
    distancias = np.sqrt(np.sum((y_real - y_pred) ** 2, axis=1))
    mae = np.mean(distancias)
    rmse = np.sqrt(np.mean(distancias ** 2))
    return distancias, mae, rmse

# Defino los modelos con los mismos hiperparámetros de la Tabla 3.2.
def crear_modelos():
    return {
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

# Entreno los modelos una sola vez, sin perturbación.
# El entrenamiento no cambia entre escenarios porque las perturbaciones
# solo afectan al conjunto de validación, simulando que el sistema
# ya está desplegado y recibe señales degradadas.
# Mido el tiempo de entrenamiento de cada modelo para comparar
# su coste computacional en la fase de ajuste.
print("Entrenando modelos...")
modelos_entrenados = {}
tiempos_train = {}
modelos = crear_modelos()

for nombre, modelo in modelos.items():
    inicio_train = time.time()
    if nombre == 'MLP':
        modelo.fit(X_train, y_train_scaled)
    else:
        modelo.fit(X_train, y_train)
    tiempos_train[nombre] = time.time() - inicio_train
    modelos_entrenados[nombre] = modelo

print("Modelos entrenados.\n")

# Evalúo cada modelo bajo los tres niveles de ruido.
# Para cada σ, aplico la ecuación 3.1 de la memoria:
#   RSSI'(i,j) = RSSI(i,j) + N(0, σ²)
# El ruido solo se añade a valores RSSI reales (distintos de 100),
# porque no tiene sentido añadir ruido a un AP que no fue detectado.
sigmas = [2, 4, 6]
resultados_ruido = {}

print("=" * 60)
print("RUIDO GAUSSIANO - Resultados")
print("=" * 60)

for sigma in sigmas:
    print(f"\n--- σ = {sigma} dBm ---")
    print(f"{'Modelo':<16} {'MAE (m)':>10} {'RMSE (m)':>10} {'Val (s)':>10}")
    print("-" * 48)

    resultados_ruido[sigma] = {}

    # Genero el ruido gaussiano con semilla fija para reproducibilidad.
    # La semilla 42 garantiza que cualquiera que ejecute este script
    # obtenga exactamente los mismos resultados.
    np.random.seed(42)
    ruido = np.random.normal(0, sigma, X_val_original.shape)
    X_val_ruidoso = X_val_original.copy()
    

    # Creo una máscara para identificar los valores RSSI reales
    # (distintos del centinela 100) y aplico el ruido solo a esos.
    mascara_activos = X_val_original != 100
    X_val_ruidoso[mascara_activos] += ruido[mascara_activos]

    # Aplico el mismo preprocesado que en el escenario base:
    # primero sustituyo el centinela 100 por -110 dBm,
    # después normalizo con el scaler ajustado en entrenamiento.
    X_val_ruidoso[X_val_ruidoso == 100] = -110

    # Mantener valores en rango válido 
    # El ruido puede producir valores fuera del rango original.
    # Los limitamos al rango observado en entrenamiento para que
    # el escalador funcione consistentemente.
    X_val_ruidoso = np.clip(X_val_ruidoso, -110, 0)

    X_val_norm = scaler_x.transform(X_val_ruidoso)

 # === DIAGNÓSTICO: DISTRIBUCIÓN DE DATOS ===
    #print(f"  Shape X_val_norm: {X_val_norm.shape}")
    #print(f"  Min X_val_norm: {X_val_norm.min():.6f}")
    #print(f"  Max X_val_norm: {X_val_norm.max():.6f}")
    #print(f"  Mean X_val_norm: {X_val_norm.mean():.6f}")
    #print(f"  Std X_val_norm: {X_val_norm.std():.6f}")
    #print(f"  NaN count: {np.isnan(X_val_norm).sum()}")
    #print(f"  Inf count: {np.isinf(X_val_norm).sum()}")
    #np.savetxt("sigma"+str(sigma)+".csv",X_val_ruidoso , delimiter=",", fmt="%d")
    #np.savetxt("normal.csv",X_val_original , delimiter=",", fmt="%d")
    #np.savetxt("ruido"+str(sigma)+".csv",ruido, delimiter=",", fmt="%d")  

    for nombre, modelo in modelos_entrenados.items():
        # Mido el tiempo de validación para cada modelo y nivel de ruido.
        inicio_val = time.time()
        if nombre == 'MLP':
             y_pred_scaled = modelo.predict(X_val_norm)
             y_pred = scaler_y.inverse_transform(y_pred_scaled)
        else:
             y_pred = modelo.predict(X_val_norm)
        tiempo_val = time.time() - inicio_val

        distancias, mae, rmse = calcular_metricas(y_val, y_pred)
        resultados_ruido[sigma][nombre] = {
             'mae': mae, 'rmse': rmse, 'distancias': distancias,
             'tiempo_train': tiempos_train[nombre],
             'tiempo_val': tiempo_val
         }

        print(f"{nombre:<16} {mae:>10.2f} {rmse:>10.2f} {tiempo_val:>10.4f}") 

# Guardo los resultados para las gráficas del script 06.
np.save('results/processed/resultados_ruido.npy', resultados_ruido, allow_pickle=True)
# Guardo los vectores de distancias individuales de KNN para el scatter plot del script 06
res_base = np.load('results/processed/resultados_base.npy', allow_pickle=True).item()
np.save('results/processed/distancias_base_knn.npy', res_base['KNN']['distancias'])
np.save('results/processed/distancias_ruido6_knn.npy', resultados_ruido[6]['KNN']['distancias'])
print("\nResultados guardados en results/processed/resultados_ruido.npy")

# === DIAGNÓSTICO: VALIDAR COHERENCIA KNN RUIDO σ=6 ===
# Verifico que los valores calculados para KNN coincidan con la tabla
# y que la dispersión de errores individuales tenga sentido.
# Esto es importante para justificar la mejora ligera de KNN bajo ruido.

print("\n" + "=" * 60)
print("DIAGNÓSTICO: KNN RUIDO σ=6 vs BASE")
print("=" * 60)

# Obtengo los vectores de distancias individuales del escenario base
# (que ya está calculado en resultados del script 03)
res_base = np.load('results/processed/resultados_base.npy', allow_pickle=True).item()
distancias_base_knn = res_base['KNN']['distancias']
distancias_ruido_knn = resultados_ruido[6]['KNN']['distancias']

print(f"\n--- LONGITUD DE VECTORES ---")
print(f"Muestras base: {len(distancias_base_knn)}")
print(f"Muestras ruido σ=6: {len(distancias_ruido_knn)}")

print(f"\n--- MEDIAS (MAE) ---")
print(f"MAE base KNN (tabla): 10,25 m")
print(f"MAE base KNN (calculado): {distancias_base_knn.mean():.2f} m")
print(f"MAE ruido σ=6 KNN (tabla): 10,07 m")
print(f"MAE ruido σ=6 KNN (calculado): {distancias_ruido_knn.mean():.2f} m")

print(f"\n--- ESTADÍSTICAS DESCRIPTIVAS ---")
print(f"{'':.<30} {'Base':>15} {'Ruido σ=6':>15}")
print(f"{'Mínimo':<30} {distancias_base_knn.min():>15.2f} {distancias_ruido_knn.min():>15.2f}")
print(f"{'Máximo':<30} {distancias_base_knn.max():>15.2f} {distancias_ruido_knn.max():>15.2f}")
print(f"{'Mediana':<30} {np.median(distancias_base_knn):>15.2f} {np.median(distancias_ruido_knn):>15.2f}")
print(f"{'Desv. Estándar':<30} {np.std(distancias_base_knn):>15.2f} {np.std(distancias_ruido_knn):>15.2f}")

print(f"\n--- ANÁLISIS DE CAMBIO POR MUESTRA ---")
n_mejoran = np.sum(distancias_ruido_knn < distancias_base_knn)
n_empeoran = np.sum(distancias_ruido_knn > distancias_base_knn)
n_igual = np.sum(distancias_ruido_knn == distancias_base_knn)
print(f"Muestras que mejoran: {n_mejoran} ({n_mejoran/len(distancias_base_knn)*100:.1f}%)")
print(f"Muestras que empeoran: {n_empeoran} ({n_empeoran/len(distancias_base_knn)*100:.1f}%)")
print(f"Muestras sin cambio: {n_igual}")

print(f"\n--- CAMBIO PROMEDIO ---")
cambio_promedio = distancias_ruido_knn.mean() - distancias_base_knn.mean()
cambio_pct = (cambio_promedio / distancias_base_knn.mean()) * 100
print(f"Cambio MAE: {cambio_promedio:+.2f} m ({cambio_pct:+.1f}%)")

print("\n" + "=" * 60)