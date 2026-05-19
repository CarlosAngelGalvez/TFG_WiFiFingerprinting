# Scripts del pipeline experimental

Esta carpeta contiene los scripts principales utilizados en el desarrollo del trabajo.

## Orden de ejecución

1. `01_carga_exploracion.py`  
   Carga inicial del dataset y exploración básica de los datos.

2. `02_preprocesado.py`  
   Sustitución de valores de ausencia de señal, normalización de variables y preparación de matrices de entrenamiento y validación.

3. `03_escenario_base.py`  
   Entrenamiento y evaluación de los modelos en el escenario base.

4. `04_ruido_gaussiano.py`  
   Evaluación de los modelos bajo perturbaciones de ruido gaussiano aplicadas al conjunto de validación.

5. `05_ap_missing.py`  
   Evaluación de los modelos bajo pérdida aleatoria de puntos de acceso.

6. `06_graficas_resultados.py`  
   Generación de las figuras utilizadas en la memoria del trabajo.