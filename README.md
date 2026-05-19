# Comparativa de modelos de aprendizaje automático para localización indoor mediante WiFi fingerprinting

Este repositorio contiene el código desarrollado para el Trabajo Final de Grado de Ingeniería de Tecnologías y Servicios de Telecomunicaciones de la Universitat Oberta de Catalunya.

El objetivo del proyecto es comparar distintos modelos de aprendizaje automático aplicados a la localización en interiores mediante WiFi fingerprinting, evaluando su precisión y robustez ante perturbaciones de señal inspiradas en entornos industriales.

## Modelos evaluados

- K-Nearest Neighbors (KNN)
- Random Forest
- Multilayer Perceptron (MLP)

## Escenarios experimentales

- Escenario base
- Ruido gaussiano sobre las medidas RSSI
- Pérdida aleatoria de puntos de acceso (AP missing)

## Métricas utilizadas

- MAE
- RMSE
- Error espacial
- Tiempo de entrenamiento
- Tiempo de validación

## Estructura del repositorio

```text
data/                 Dataset original utilizado en el estudio
src/                  Scripts principales del pipeline experimental
results/figures/      Figuras generadas para la memoria
results/processed/    Resultados y archivos procesados
requirements.txt      Dependencias necesarias
```

## Dataset

El proyecto utiliza el dataset público UJIIndoorLoc.

Los archivos esperados son:

```text
trainingData.csv
validationData.csv
```

Estos archivos deben ubicarse en la carpeta `data/`.

## Instalación

Crear un entorno virtual:

```bash
python -m venv venv
```

Activar el entorno en Windows:

```bash
venv\Scripts\activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución

Los scripts deben ejecutarse desde la raíz del proyecto en el siguiente orden:

```bash
python src/01_carga_exploracion.py
python src/02_preprocesado.py
python src/03_escenario_base.py
python src/04_ruido_gaussiano.py
python src/05_ap_missing.py
python src/06_graficas_resultados.py
```

## Resultados

Las figuras generadas se almacenan en la carpeta:

```text
results/figures/
```

Los archivos procesados y resultados intermedios se almacenan en:

```text
results/processed/
```

## Autor

Carlos Ángel Gálvez  
Trabajo Final de Grado 
Grado en Ing. de Tecnologías y Servicios de Telecomunicación
Universitat Oberta de Catalunya
