# 01_carga_exploracion.py
# Este script carga el dataset UJIIndoorLoc y muestra un resumen
# de sus características principales para verificar que coinciden
# con lo descrito en la Tabla 3.1 de la memoria del TFG.

import pandas as pd
import numpy as np

# Cargo los dos archivos CSV del dataset: entrenamiento y validación.
# Son las particiones originales que proporciona UJIIndoorLoc,
# así que no hace falta dividir nada manualmente.
df_train = pd.read_csv('data/trainingData.csv')
df_val = pd.read_csv('data/validationData.csv')

# Compruebo las dimensiones de ambos conjuntos.
# Espero 19937 muestras de entrenamiento y 1111 de validación,
# cada una con 529 columnas (520 WAPs + 9 variables auxiliares).
print("=" * 60)
print("DIMENSIONES DEL DATASET")
print("=" * 60)
print(f"Entrenamiento: {df_train.shape[0]} muestras, {df_train.shape[1]} columnas")
print(f"Validación:    {df_val.shape[0]} muestras, {df_val.shape[1]} columnas")

# Identifico las columnas correspondientes a los puntos de acceso.
# En el dataset, van de WAP001 a WAP520 y contienen los valores RSSI.
wap_cols = [col for col in df_train.columns if col.startswith('WAP')]
print(f"\nPuntos de acceso (WAP): {len(wap_cols)}")

# Analizo el rango de valores RSSI reales, excluyendo el valor 100
# que es el centinela que usa UJIIndoorLoc para indicar que un AP
# no fue detectado en esa medición.
rssi_train = df_train[wap_cols].values
rssi_reales = rssi_train[rssi_train != 100]
print(f"\nRango RSSI observado (sin centinela): {rssi_reales.min()} a {rssi_reales.max()} dBm")

# Calculo qué proporción de la matriz RSSI son valores de ausencia (100).
# Este dato confirma la naturaleza dispersa del dataset que comento
# en el apartado 3.2 de la memoria.
total_valores = rssi_train.size
valores_ausentes = (rssi_train == 100).sum()
porcentaje = (valores_ausentes / total_valores) * 100
print(f"\nValores de ausencia (100): {valores_ausentes:,} de {total_valores:,} ({porcentaje:.1f}%)")

# Reviso el rango de las coordenadas espaciales (variables objetivo).
# Están en sistema UTM, expresadas en metros, lo que permite calcular
# directamente la distancia euclídea sin conversiones.
print(f"\nVariables objetivo disponibles:")
print(f"  LONGITUDE: {df_train['LONGITUDE'].min():.2f} a {df_train['LONGITUDE'].max():.2f}")
print(f"  LATITUDE:  {df_train['LATITUDE'].min():.2f} a {df_train['LATITUDE'].max():.2f}")

# Compruebo cuántos edificios y plantas hay en el dataset.
# Según la documentación de Torres-Sospedra et al. (2014),
# son 3 edificios con entre 4 y 5 plantas cada uno.
print(f"\nEdificios: {df_train['BUILDINGID'].nunique()}")
print(f"Plantas por edificio:")
for bid in sorted(df_train['BUILDINGID'].unique()):
    plantas = df_train[df_train['BUILDINGID'] == bid]['FLOOR'].nunique()
    print(f"  Edificio {bid}: {plantas} plantas")

# Resumen final para contrastar con la Tabla 3.1 de la memoria.
print("\n" + "=" * 60)
print("RESUMEN - Verificación contra Tabla 3.1 de la memoria")
print("=" * 60)
print(f"Muestras entrenamiento:  {df_train.shape[0]}")
print(f"Muestras validación:     {df_val.shape[0]}")
print(f"Puntos de acceso (WAP):  {len(wap_cols)}")
print(f"Rango RSSI:              {rssi_reales.min()} a {rssi_reales.max()} dBm")
print(f"Valor centinela:         100")
print(f"Edificios:               {df_train['BUILDINGID'].nunique()}")