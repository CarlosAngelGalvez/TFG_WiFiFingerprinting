# 02_preprocesado.py
# Este script aplica el preprocesado descrito en el apartado 3.2 de la memoria:
# 1) Sustituir el valor centinela 100 por -110 dBm
# 2) Normalizar las variables RSSI con Min-Max
# 3) Escalar las variables objetivo para que el MLP converja correctamente
# Al final guarda todo en archivos .npy para usarlos en los scripts siguientes.

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
from pathlib import Path

# Rutas del proyecto
DATA_DIR = Path("data")
PROCESSED_DIR = Path("results") / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Cargo los CSV originales del dataset.
df_train = pd.read_csv(DATA_DIR / "trainingData.csv")
df_val = pd.read_csv(DATA_DIR / "validationData.csv")

# Separo las columnas de los puntos de acceso (WAP001-WAP520)
# y las variables objetivo (LONGITUDE, LATITUDE).
# El resto de columnas (edificio, planta, usuario, etc.) no se usan
# como entrada del modelo, tal como explico en el apartado 3.2.
wap_cols = [col for col in df_train.columns if col.startswith('WAP')]
target_cols = ['LONGITUDE', 'LATITUDE']

# Extraigo las matrices numéricas para trabajar con NumPy.
# Las convierto a float para evitar problemas con operaciones posteriores.
X_train_raw = df_train[wap_cols].values.astype(float)
X_val_raw = df_val[wap_cols].values.astype(float)
y_train = df_train[target_cols].values
y_val = df_val[target_cols].values

print("=" * 60)
print("PREPROCESADO")
print("=" * 60)

# PASO 1: Sustituyo el valor centinela 100 por -110 dBm.
# El dataset usa 100 para indicar que un AP no fue detectado,
# pero ese valor queda fuera del rango real de RSSI (-104 a 0 dBm)
# y distorsionaría el entrenamiento. Uso -110 porque está por debajo
# del mínimo observado (-104), manteniendo la coherencia del rango.
X_train_raw[X_train_raw == 100] = -110
X_val_raw[X_val_raw == 100] = -110

print(f"Centinela sustituido: 100 → -110 dBm")
print(f"Rango tras sustitución (train): {X_train_raw.min()} a {X_train_raw.max()} dBm")
print(f"Rango tras sustitución (val):   {X_val_raw.min()} a {X_val_raw.max()} dBm")

# PASO 2: Normalización Min-Max de las variables de entrada.
# El scaler se ajusta SOLO con los datos de entrenamiento (fit_transform)
# y se aplica a validación con transform, para no filtrar información
# del conjunto de validación al entrenamiento.

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)

print(f"\nNormalización Min-Max aplicada")
print(f"Rango normalizado (train): {X_train.min():.2f} a {X_train.max():.2f}")
print(f"Rango normalizado (val):   {X_val.min():.4f} a {X_val.max():.4f}")

# PASO 3: Escalado de las variables objetivo con StandardScaler.
# Esto es necesario para que el MLP converja correctamente, ya que
# las coordenadas UTM tienen valores muy grandes (del orden de millones
# en latitud) y el optimizador Adam tiene dificultades con esas magnitudes.
# KNN y Random Forest no necesitan este escalado, pero no les afecta
# porque les paso las coordenadas originales directamente.
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)

# Verifico las dimensiones finales de todas las matrices.
print(f"\nFormas finales:")
print(f"  X_train: {X_train.shape}")
print(f"  X_val:   {X_val.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  y_val:   {y_val.shape}")

# Guardo todo en archivos para que los scripts siguientes
# no tengan que repetir el preprocesado cada vez.
# Los .npy son matrices NumPy y los .pkl son objetos de scikit-learn
# (los scalers, que necesitaré para transformar/destransformar datos).
np.save(PROCESSED_DIR / "X_train.npy", X_train)
np.save(PROCESSED_DIR / "X_val.npy", X_val)
np.save(PROCESSED_DIR / "y_train.npy", y_train)
np.save(PROCESSED_DIR / "y_val.npy", y_val)
np.save(PROCESSED_DIR / "y_train_scaled.npy", y_train_scaled)

with open(PROCESSED_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open(PROCESSED_DIR / "scaler_y.pkl", "wb") as f:
    pickle.dump(scaler_y, f)

print(f"\nArchivos guardados en: {PROCESSED_DIR}")
print("  X_train.npy")
print("  X_val.npy")
print("  y_train.npy")
print("  y_val.npy")
print("  y_train_scaled.npy")
print("  scaler.pkl")
print("  scaler_y.pkl")
print("Preprocesado completado.")