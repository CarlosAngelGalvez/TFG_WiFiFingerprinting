# 06_graficas_resultados.py
# Este script genera todas las visualizaciones y tablas resumen
# para el Capítulo 4 de la memoria. Carga los resultados guardados
# por los scripts 03, 04 y 05 y produce:
# - Tabla resumen con MAE y RMSE de todos los escenarios
# - Tabla de degradación porcentual respecto al escenario base
# - Gráfica de barras del escenario base
# - Gráficas de líneas de evolución bajo ruido y AP missing
# - Heatmap de degradación
# - Gráfica de barras del escenario más extremo (AP missing 30%)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

PROCESSED_DIR = Path("results") / "processed"
FIGURES_DIR = Path("results") / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Cargo los resultados de los tres escenarios.
# Cada archivo .npy contiene un diccionario con las métricas
# de cada modelo, generado por los scripts anteriores.
res_base = np.load(PROCESSED_DIR / 'resultados_base.npy', allow_pickle=True).item()
res_ruido = np.load(PROCESSED_DIR / 'resultados_ruido.npy', allow_pickle=True).item()
res_missing = np.load(PROCESSED_DIR / 'resultados_missing.npy', allow_pickle=True).item()

modelos = ['KNN', 'Random Forest', 'MLP']

# === TABLA RESUMEN ===
# Esta tabla contiene todos los valores MAE/RMSE y es la base
# para construir las tablas LaTeX del Capítulo 4.
print("=" * 70)
print("TABLA RESUMEN - Todos los escenarios")
print("=" * 70)
print(f"{'Escenario':<20} {'KNN':>14} {'RF':>14} {'MLP':>14}")
print(f"{'':.<20} {'MAE / RMSE':>14} {'MAE / RMSE':>14} {'MAE / RMSE':>14}")
print("-" * 70)

fila = f"{'Base':<20}"
for m in modelos:
    fila += f" {res_base[m]['mae']:>6.2f}/{res_base[m]['rmse']:<6.2f}"
print(fila)

for sigma in [2, 4, 6]:
    fila = f"{'Ruido σ=' + str(sigma):<20}"
    for m in modelos:
        fila += f" {res_ruido[sigma][m]['mae']:>6.2f}/{res_ruido[sigma][m]['rmse']:<6.2f}"
    print(fila)

for p in [10, 20, 30]:
    fila = f"{'AP missing ' + str(p) + '%':<20}"
    for m in modelos:
        fila += f" {res_missing[p][m]['mae']:>6.2f}/{res_missing[p][m]['rmse']:<6.2f}"
    print(fila)

# === TABLA DE DEGRADACIÓN PORCENTUAL ===
# Calculo cuánto empeora el MAE de cada modelo respecto al escenario base.
# Esto permite ver de un vistazo quién se degrada más en cada escenario,
# independientemente de si su MAE base era mejor o peor.
print("\n" + "=" * 70)
print("DEGRADACIÓN PORCENTUAL DEL MAE respecto al escenario base")
print("=" * 70)
print(f"{'Escenario':<20} {'KNN':>10} {'RF':>10} {'MLP':>10}")
print("-" * 52)

for sigma in [2, 4, 6]:
    fila = f"{'Ruido σ=' + str(sigma):<20}"
    for m in modelos:
        deg = ((res_ruido[sigma][m]['mae'] - res_base[m]['mae']) / res_base[m]['mae']) * 100
        fila += f" {deg:>+9.1f}%"
    print(fila)

for p in [10, 20, 30]:
    fila = f"{'AP missing ' + str(p) + '%':<20}"
    for m in modelos:
        deg = ((res_missing[p][m]['mae'] - res_base[m]['mae']) / res_base[m]['mae']) * 100
        fila += f" {deg:>+9.1f}%"
    print(fila)

# === GRÁFICA 1: BARRAS DEL ESCENARIO BASE ===
# Comparo visualmente los tres modelos en condiciones sin perturbación.
# Uso barras dobles (MAE y RMSE) para cada modelo.
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(modelos))
ancho = 0.3

maes_base = [res_base[m]['mae'] for m in modelos]
rmses_base = [res_base[m]['rmse'] for m in modelos]

barras_mae = ax.bar(x - ancho/2, maes_base, ancho, label='MAE', color='#2196F3')
barras_rmse = ax.bar(x + ancho/2, rmses_base, ancho, label='RMSE', color='#FF9800')

# Pongo los valores encima de cada barra para facilitar la lectura.
for barra in barras_mae:
    ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.3,
            f'{barra.get_height():.2f}', ha='center', va='bottom', fontsize=9)
for barra in barras_rmse:
    ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.3,
            f'{barra.get_height():.2f}', ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Error (m)', fontsize=12)
ax.set_title('Rendimiento en el escenario base', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(modelos, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_barras_base.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nGráfica guardada: results/figures/fig_barras_base.png")

# === GRÁFICA 2: LÍNEAS MAE BAJO AMBAS PERTURBACIONES ===
# Panel izquierdo: evolución del MAE con ruido gaussiano creciente.
# Panel derecho: evolución del MAE con pérdida progresiva de APs.
# En ambos casos el punto de partida es el escenario base.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sigmas = [0, 2, 4, 6]
for m in modelos:
    maes = [res_base[m]['mae']]
    maes += [res_ruido[s][m]['mae'] for s in [2, 4, 6]]
    axes[0].plot(sigmas, maes, 'o-', label=m, linewidth=2, markersize=8)

axes[0].set_xlabel('σ (dBm)', fontsize=12)
axes[0].set_ylabel('MAE (m)', fontsize=12)
axes[0].set_title('Impacto del ruido gaussiano en el MAE', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(sigmas)
axes[0].set_xticklabels(['Base', 'σ=2', 'σ=4', 'σ=6'])

porcentajes = [0, 10, 20, 30]
for m in modelos:
    maes = [res_base[m]['mae']]
    maes += [res_missing[p][m]['mae'] for p in [10, 20, 30]]
    axes[1].plot(porcentajes, maes, 's-', label=m, linewidth=2, markersize=8)

axes[1].set_xlabel('APs eliminados (%)', fontsize=12)
axes[1].set_ylabel('MAE (m)', fontsize=12)
axes[1].set_title('Impacto de AP missing en el MAE', fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(porcentajes)
axes[1].set_xticklabels(['Base', '10%', '20%', '30%'])

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_mae_perturbaciones.png', dpi=300, bbox_inches='tight')
plt.show()
print("Gráfica guardada: results/figures/fig_mae_perturbaciones.png")

# === GRÁFICA 3: LÍNEAS RMSE BAJO AMBAS PERTURBACIONES ===
# Misma estructura que la gráfica de MAE pero con RMSE.
# El RMSE penaliza más los errores grandes, así que si un modelo
# tiene un RMSE mucho mayor que su MAE, significa que produce
# fallos puntuales severos en algunas muestras.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for m in modelos:
    rmses = [res_base[m]['rmse']]
    rmses += [res_ruido[s][m]['rmse'] for s in [2, 4, 6]]
    axes[0].plot(sigmas, rmses, 'o-', label=m, linewidth=2, markersize=8)

axes[0].set_xlabel('σ (dBm)', fontsize=12)
axes[0].set_ylabel('RMSE (m)', fontsize=12)
axes[0].set_title('Impacto del ruido gaussiano en el RMSE', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(sigmas)
axes[0].set_xticklabels(['Base', 'σ=2', 'σ=4', 'σ=6'])

for m in modelos:
    rmses = [res_base[m]['rmse']]
    rmses += [res_missing[p][m]['rmse'] for p in [10, 20, 30]]
    axes[1].plot(porcentajes, rmses, 's-', label=m, linewidth=2, markersize=8)

axes[1].set_xlabel('APs eliminados (%)', fontsize=12)
axes[1].set_ylabel('RMSE (m)', fontsize=12)
axes[1].set_title('Impacto de AP missing en el RMSE', fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(porcentajes)
axes[1].set_xticklabels(['Base', '10%', '20%', '30%'])

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_rmse_perturbaciones.png', dpi=300, bbox_inches='tight')
plt.show()
print("Gráfica guardada: results/figures/fig_rmse_perturbaciones.png")

# === GRÁFICA 4: HEATMAP DE DEGRADACIÓN ===
# Matriz coloreada donde cada celda muestra el MAE de un modelo
# en un escenario concreto. Los colores van de verde (poco error)
# a rojo (mucho error), lo que permite identificar de un vistazo
# qué combinaciones modelo-escenario son las más problemáticas.
escenarios = ['Base', 'Ruido σ=2', 'Ruido σ=4', 'Ruido σ=6',
              'AP missing 10%', 'AP missing 20%', 'AP missing 30%']

# Construyo la matriz de valores MAE (filas = escenarios, columnas = modelos).
matriz = []
matriz.append([res_base[m]['mae'] for m in modelos])
for sigma in [2, 4, 6]:
    matriz.append([res_ruido[sigma][m]['mae'] for m in modelos])
for p in [10, 20, 30]:
    matriz.append([res_missing[p][m]['mae'] for m in modelos])
matriz = np.array(matriz)

fig, ax = plt.subplots(figsize=(8, 6))

# Uso un colormap que vaya de verde (bueno) a rojo (malo).
cmap = matplotlib.colormaps['RdYlGn_r']
im = ax.imshow(matriz, cmap=cmap, aspect='auto', vmin=8, vmax=35)

# Pongo el valor numérico dentro de cada celda.
for i in range(len(escenarios)):
    for j in range(len(modelos)):
        color_texto = 'white' if matriz[i, j] > 25 else 'black'
        ax.text(j, i, f'{matriz[i, j]:.1f}', ha='center', va='center',
                fontsize=11, fontweight='bold', color=color_texto)

ax.set_xticks(range(len(modelos)))
ax.set_xticklabels(modelos, fontsize=11)
ax.set_yticks(range(len(escenarios)))
ax.set_yticklabels(escenarios, fontsize=10)
ax.set_title('MAE (m) por modelo y escenario', fontsize=13)

# Barra de color lateral para referencia.
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('MAE (m)', fontsize=11)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_heatmap_mae.png', dpi=300, bbox_inches='tight')
plt.show()
print("Gráfica guardada: results/figures/fig_heatmap_mae.png")

# === GRÁFICA 5: BARRAS AP MISSING 30% ===
# Este es el escenario más extremo del estudio. Las diferencias
# entre modelos son muy grandes aquí, así que la gráfica de barras
# tiene un impacto visual potente que refuerza las conclusiones.
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(modelos))
ancho = 0.3

maes_30 = [res_missing[30][m]['mae'] for m in modelos]
rmses_30 = [res_missing[30][m]['rmse'] for m in modelos]

barras_mae = ax.bar(x - ancho/2, maes_30, ancho, label='MAE', color='#F44336')
barras_rmse = ax.bar(x + ancho/2, rmses_30, ancho, label='RMSE', color='#9C27B0')

for barra in barras_mae:
    ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.5,
            f'{barra.get_height():.2f}', ha='center', va='bottom', fontsize=9)
for barra in barras_rmse:
    ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.5,
            f'{barra.get_height():.2f}', ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Error (m)', fontsize=12)
ax.set_title('Rendimiento con AP missing al 30%', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(modelos, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_barras_ap30.png', dpi=300, bbox_inches='tight')
plt.show()
print("Gráfica guardada: results/figures/fig_barras_ap30.png")

print("\n" + "=" * 70)
print("Todas las gráficas generadas correctamente.")
print("Archivos guardados en results/figures/")
print("=" * 70)

# === GRÁFICA 6: BOXPLOT DE ERRORES BASE vs AP MISSING 30% ===
# El boxplot muestra la distribución completa de los errores espaciales,
# no solo la media (MAE) y la raíz cuadrática (RMSE).
# Esto permite ver la mediana, los cuartiles y los outliers de cada modelo.
# Comparo el escenario base con AP missing 30% para visualizar
# cómo cambia la dispersión de los errores en el caso más extremo.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel izquierdo: escenario base
datos_base = [res_base[m]['distancias'] for m in modelos]
bp1 = axes[0].boxplot(datos_base, labels=modelos, patch_artist=True,
                       showfliers=True, flierprops=dict(marker='o', markersize=3, alpha=0.3))
colores = ['#2196F3', '#FF9800', '#4CAF50']
for patch, color in zip(bp1['boxes'], colores):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[0].set_ylabel('Error espacial (m)', fontsize=12)
axes[0].set_title('Distribución del error - Escenario base', fontsize=13)
axes[0].grid(axis='y', alpha=0.3)

# Panel derecho: AP missing 30%
datos_ap30 = [res_missing[30][m]['distancias'] for m in modelos]
bp2 = axes[1].boxplot(datos_ap30, labels=modelos, patch_artist=True,
                       showfliers=True, flierprops=dict(marker='o', markersize=3, alpha=0.3))
for patch, color in zip(bp2['boxes'], colores):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1].set_ylabel('Error espacial (m)', fontsize=12)
axes[1].set_title('Distribución del error - AP missing 30%', fontsize=13)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_boxplot_errores.png', dpi=300, bbox_inches='tight')
plt.show()
print("Gráfica guardada: results/figures/fig_boxplot_errores.png")

# === GRÁFICA 7: SCATTER POSICIONES REALES vs ESTIMADAS (MLP) ===
# Muestro las coordenadas reales y las estimadas por el MLP
# en el escenario base y en AP missing 30%.
# Uso solo MLP porque es el modelo más robusto del estudio
# y permite ver cómo se degrada la estimación espacial
# cuando se pierde un 30% de la infraestructura.
# Necesito recalcular las predicciones del MLP para ambos escenarios.

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd

X_train = np.load(PROCESSED_DIR / 'X_train.npy')
y_train = np.load(PROCESSED_DIR / 'y_train.npy')
y_val = np.load(PROCESSED_DIR / 'y_val.npy')
y_train_scaled = np.load(PROCESSED_DIR / 'y_train_scaled.npy')

import pickle
with open(PROCESSED_DIR / 'scaler.pkl', 'rb') as f:
    scaler_x = pickle.load(f)
with open(PROCESSED_DIR / 'scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

# Entreno el MLP una vez.
mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                   max_iter=500, early_stopping=True, n_iter_no_change=10, random_state=42)
mlp.fit(X_train, y_train_scaled)

# Predicción escenario base.
X_val_base = np.load(PROCESSED_DIR / 'X_val.npy')
y_pred_base_scaled = mlp.predict(X_val_base)
y_pred_base = scaler_y.inverse_transform(y_pred_base_scaled)

# Predicción AP missing 30%.
df_val = pd.read_csv('data/validationData.csv')
wap_cols = [col for col in df_val.columns if col.startswith('WAP')]
X_val_original = df_val[wap_cols].values.astype(float)
n_waps = len(wap_cols)

np.random.seed(42)
n_eliminar = int(n_waps * 30 / 100)
aps_eliminados = np.random.choice(n_waps, size=n_eliminar, replace=False)
X_val_missing = X_val_original.copy()
X_val_missing[:, aps_eliminados] = 100
X_val_missing[X_val_missing == 100] = -110
X_val_missing_norm = scaler_x.transform(X_val_missing)

y_pred_ap30_scaled = mlp.predict(X_val_missing_norm)
y_pred_ap30 = scaler_y.inverse_transform(y_pred_ap30_scaled)

# Dibujo los scatter.
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel izquierdo: escenario base.
axes[0].scatter(y_val[:, 0], y_val[:, 1], s=8, alpha=0.5, label='Posición real', color='#2196F3')
axes[0].scatter(y_pred_base[:, 0], y_pred_base[:, 1], s=8, alpha=0.5, label='Estimación MLP', color='#F44336')
axes[0].set_xlabel('Coordenada X (m)', fontsize=11)
axes[0].set_ylabel('Coordenada Y (m)', fontsize=11)
axes[0].set_title('MLP - Escenario base', fontsize=13)
axes[0].legend(fontsize=10, markerscale=3)
axes[0].grid(True, alpha=0.2)

# Panel derecho: AP missing 30%.
axes[1].scatter(y_val[:, 0], y_val[:, 1], s=8, alpha=0.5, label='Posición real', color='#2196F3')
axes[1].scatter(y_pred_ap30[:, 0], y_pred_ap30[:, 1], s=8, alpha=0.5, label='Estimación MLP', color='#F44336')
axes[1].set_xlabel('Coordenada X (m)', fontsize=11)
axes[1].set_ylabel('Coordenada Y (m)', fontsize=11)
axes[1].set_title('MLP - AP missing 30%', fontsize=13)
axes[1].legend(fontsize=10, markerscale=3)
axes[1].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_scatter_mlp.png', dpi=300, bbox_inches='tight')
plt.show()
print("Gráfica guardada: results/figures/fig_scatter_mlp.png")

# === GRÁFICA 8: TIEMPOS DE ENTRENAMIENTO Y VALIDACIÓN ===
# Comparo el coste computacional de cada modelo en el escenario base.
# Esta gráfica complementa las métricas de precisión con información
# sobre la viabilidad práctica de cada modelo, aspecto relevante
# para despliegues en entornos con recursos de cómputo limitados.
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(modelos))
ancho = 0.3

tiempos_train = [res_base[m]['tiempo_train'] for m in modelos]
tiempos_val = [res_base[m]['tiempo_val'] for m in modelos]

barras_train = ax.bar(x - ancho/2, tiempos_train, ancho,
                       label='Entrenamiento', color='#2196F3')
barras_val = ax.bar(x + ancho/2, tiempos_val, ancho,
                     label='Validación', color='#FF9800')

for barra in barras_train:
    ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.1,
            f'{barra.get_height():.2f}', ha='center', va='bottom', fontsize=9)
for barra in barras_val:
    ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.1,
            f'{barra.get_height():.4f}', ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Tiempo (s)', fontsize=12)
ax.set_title('Tiempos de entrenamiento y validación - Escenario base', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(modelos, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_tiempos.png', dpi=300, bbox_inches='tight')
plt.show()
print("Gráfica guardada: results/figures/fig_tiempos.png")