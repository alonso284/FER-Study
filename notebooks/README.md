# ML - Documentacion de sesiones (basada en notebooks)

## Descripcion general
Este repositorio contiene notebooks de practicas guiadas de Machine Learning, junto con actividades autonomas y una evidencia integradora. El flujo recorre fundamentos, regresion, clasificacion, redes neuronales, clustering, series de tiempo y un mini proyecto de CNN con MNIST.

## Estructura del repositorio

| Notebook | Sesion / Tema principal | Tipo de problema |
|---|---|---|
| `Sesion1_Fundamentos_ML_Notebook.ipynb` | Sesion 1 - Fundamentos de ML | Regresion supervisada |
| `Notebook_regresion_lineal_guiada.ipynb` | Sesion 2 - Regresion lineal guiada | Regresion |
| `Notebook_regresion_logistica_guiada.ipynb` | Sesion 2 - Regresion logistica guiada | Clasificacion binaria |
| `Notebook_red_sencilla_sesion5.ipynb` | Sesion 5 - Red neuronal sencilla (MLP) | Clasificacion binaria |
| `Notebook_clustering_PCA_iris_pokemon.ipynb` | Sesion 7 - Clustering + PCA | No supervisado |
| `Notebook_clasificacion_iris_pokemon.ipynb` | Sesion 8 - Clasificacion (Iris y Pokemon) | Clasificacion multiclase/binaria |
| `Notebook_AR_MA_sesion.ipynb` | Modelos AR(1) y MA(1) | Series de tiempo |
| `Mini_CNN_Workbook_Local_MNIST.ipynb` | Mini CNN con TensorFlow (MNIST local) | Clasificacion de imagenes |
| `EvidenciaIA_Seleccion_Modelos.ipynb` | Evidencia integradora - Seleccion y evaluacion de modelos | Segun dataset |
| `A00836072.ipynb` | Evidencia aplicada - Cancer de mama (comparacion de modelos) | Clasificacion binaria |
| `tarea3.ipynb` | Tarea 3 - Simulacion (transformada inversa e importance sampling) | Simulacion/Metodos numericos |

---

## Documentacion por sesion

### 1) Sesion 1 - Fundamentos de ML
**Notebook:** `Sesion1_Fundamentos_ML_Notebook.ipynb`

**Objetivo**
- Entender el pipeline basico de Machine Learning.
- Identificar features y target.
- Entrenar y evaluar un primer modelo.

**Contenidos clave**
- Carga y exploracion de datos (`tips.csv`).
- Separacion train/test.
- Entrenamiento con `LinearRegression`.
- Predicciones y visualizacion.
- Experimento con variables categoricas y `get_dummies`.

**Habilidades trabajadas**
- Formular un problema de regresion.
- Interpretar salidas basicas del modelo.
- Entender impacto de la preparacion de datos.

---

### 2) Sesion 2 - Regresion lineal guiada
**Notebook:** `Notebook_regresion_lineal_guiada.ipynb`

**Objetivo**
- Conectar el procedimiento algebraico manual con la implementacion en Python.

**Contenidos clave**
- Modelo lineal con dos variables: $y = a_1x_1 + a_2x_2 + b$.
- Entrenamiento con `LinearRegression`.
- Interpretacion de coeficientes e intercepto.
- Evaluacion con MAE, MSE y $R^2$.
- Analisis del error al introducir datos menos perfectos.

**Habilidades trabajadas**
- Interpretar parametros del modelo.
- Relacionar calidad de datos con error del modelo.

---

### 3) Sesion 2 - Regresion logistica guiada
**Notebook:** `Notebook_regresion_logistica_guiada.ipynb`

**Objetivo**
- Comparar una regla de clasificacion manual contra un modelo de regresion logistica.

**Contenidos clave**
- Regla manual con umbral sobre una combinacion lineal ($z$).
- Clasificacion binaria con `LogisticRegression`.
- Probabilidades (`predict_proba`) y clase final.
- Matriz de confusion, accuracy y reporte de clasificacion.

**Habilidades trabajadas**
- Diferenciar score/probabilidad vs clase.
- Entender por que regresion logistica se usa en clasificacion.

---

### 4) Sesion 5 - Red neuronal sencilla
**Notebook:** `Notebook_red_sencilla_sesion5.ipynb`

**Objetivo**
- Entrenar una red neuronal pequena y comparar arquitecturas.

**Contenidos clave**
- Dataset sintetico de clasificacion (2 entradas, salida binaria).
- `MLPClassifier` con capa oculta pequena.
- Prediccion, accuracy, matriz de confusion y reporte.
- Inspeccion de pesos y biases.
- Prediccion sobre casos nuevos con probabilidades.
- Comparacion entre 2, 3 o 4 neuronas ocultas.

**Habilidades trabajadas**
- Relacionar arquitectura y desempeno.
- Entender sobreajuste en modelos mas grandes.

---

### 5) Sesion 7 - Clustering + PCA
**Notebook:** `Notebook_clustering_PCA_iris_pokemon.ipynb`

**Objetivo**
- Aplicar aprendizaje no supervisado y visualizar resultados.

**Contenidos clave**
- K-means sobre Iris.
- Estandarizacion de variables.
- Metodos para elegir K: codo e `silhouette score`.
- PCA para reduccion a 2D.
- Comparacion visual clusters vs etiquetas reales.
- Parte autonoma con dataset de Pokemon.

**Habilidades trabajadas**
- Elegir K con criterios cuantitativos.
- Interpretar perdida/ganancia al reducir dimensiones.

---

### 6) Sesion 8 - Clasificacion (Iris y Pokemon)
**Notebook:** `Notebook_clasificacion_iris_pokemon.ipynb`

**Objetivo**
- Entrenar y comparar clasificadores clasicos en dos datasets.

**Contenidos clave**
- Modelos: kNN, Arbol de decision, Naive Bayes y SVM.
- Escalamiento para modelos sensibles a distancia.
- Evaluacion con accuracy, matriz de confusion, precision, recall y F1.
- Tabla comparativa y grafico de desempeno.
- Parte autonoma: clasificacion de Pokemon (por ejemplo, `Legendary`).

**Habilidades trabajadas**
- Seleccionar modelo segun interpretabilidad y rendimiento.
- Entender limitaciones de accuracy en clases desbalanceadas.

---

### 7) Sesion de series de tiempo - AR y MA
**Notebook:** `Notebook_AR_MA_sesion.ipynb`

**Objetivo**
- Implementar manualmente modelos AR(1) y MA(1).

**Contenidos clave**
- Serie temporal simple.
- Prediccion AR(1): $y_t = c + \phi y_{t-1}$.
- Prediccion MA(1): $y_t = \mu + \theta e_{t-1}$.
- Construccion de tablas comparativas y visualizaciones.

**Habilidades trabajadas**
- Distinguir uso de valores pasados vs errores pasados.
- Comprender efecto de parametros ($\phi$, $c$, $\theta$, $\mu$).

---

### 8) Mini CNN con MNIST local
**Notebook:** `Mini_CNN_Workbook_Local_MNIST.ipynb`

**Objetivo**
- Construir, entrenar y evaluar una CNN basica para digitos MNIST.

**Contenidos clave**
- Carga local de `mnist.npz`.
- Normalizacion y reshape de imagenes.
- Arquitectura CNN (Conv2D, MaxPooling, Dense).
- Entrenamiento y evaluacion final.
- Grafica de accuracy train/validation.
- Actividad: modificar arquitectura y epocas para comparar resultados.

**Habilidades trabajadas**
- Pipeline completo de Deep Learning en clasificacion de imagenes.
- Diagnostico de overfitting comparando accuracy de entrenamiento y validacion.

---

### 9) Evidencia integradora - Seleccion y evaluacion de modelos
**Notebook:** `EvidenciaIA_Seleccion_Modelos.ipynb`

**Objetivo**
- Resolver un caso completo de ML de extremo a extremo con justificacion tecnica.

**Contenidos clave**
- Comprension del problema y formulacion del objetivo.
- Carga y exploracion de dataset.
- Limpieza/preparacion.
- Visualizaciones.
- Split train/test.
- Entrenamiento y comparacion de modelos.
- Conclusiones y recomendacion del mejor modelo.

**Entregables esperados**
- Hoja de analisis (manual).
- Notebook completo con evidencia tecnica.

---

### 10) Evidencia aplicada - Cancer de mama
**Notebook:** `A00836072.ipynb`

**Objetivo**
- Resolver una clasificacion binaria de cancer de mama y comparar modelos.

**Contenidos clave**
- Dataset `load_breast_cancer` de scikit-learn.
- Exploracion: dimensiones, variables, faltantes, duplicados, outliers.
- Split train/test.
- Entrenamiento de modelos: Regresion Logistica, kNN, Naive Bayes, SVM.
- Evaluacion con accuracy, matriz de confusion y `classification_report`.
- Tabla final comparativa de resultados.

**Habilidades trabajadas**
- Comparacion objetiva de clasificadores en un problema real.
- Uso correcto de escalamiento para modelos sensibles.

---

### 11) Tarea 3 - Simulacion y Monte Carlo
**Notebook:** `tarea3.ipynb`

**Objetivo**
- Aplicar tecnicas de simulacion para estimar distribuciones y probabilidades raras.

**Contenidos clave**
- Generacion de muestras con transformada inversa.
- Comparacion histograma empirico vs densidad teorica.
- Monte Carlo estandar para eventos raros.
- Importance Sampling para reducir varianza con menos muestras.
- Diagnostico de pesos y calculo de Effective Sample Size (ESS).

**Habilidades trabajadas**
- Elegir estrategia de estimacion segun dificultad del evento.
- Interpretar estabilidad y eficiencia de estimadores.

---

## Archivos de apoyo y datos

- `mnist.npz`: dataset local requerido para la practica CNN.
- `pokemon.csv`: dataset usado en sesiones de clasificacion y clustering (parte autonoma).
- `p1_histograma.png`, `p3_histograma.png`: salidas graficas de ejercicios de simulacion.

## Requisitos sugeridos

Dependencias principales usadas en los notebooks:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow`
- `scipy`
- `seaborn` (en algunos flujos)
- `kagglehub` (opcional, para descarga de dataset)

Instalacion sugerida:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow scipy seaborn kagglehub
```

## Recomendaciones de ejecucion

1. Ejecuta cada notebook en orden de celdas.
2. Verifica primero que los archivos de datos locales existan (`mnist.npz`, `pokemon.csv`).
3. Si cambias features/targets, documenta el cambio y vuelve a correr la evaluacion completa.
4. En comparaciones de modelos, no uses solo accuracy cuando haya desbalance de clases.

## Ruta de aprendizaje recomendada

1. Sesion 1 (fundamentos)
2. Sesion 2 (regresion lineal + logistica)
3. Sesion 5 (red neuronal sencilla)
4. Sesion 8 (clasificacion comparativa)
5. Sesion 7 (clustering + PCA)
6. AR/MA (series de tiempo)
7. Mini CNN (vision)
8. Evidencias y tareas de cierre (`A00836072`, `EvidenciaIA_Seleccion_Modelos`, `tarea3`)
