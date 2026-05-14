# Proyecto 1 - Detección de contaminaciones en imágenes mediante aprendizaje automático clásico

## Descripción

Este proyecto tiene como objetivo detectar contaminaciones en una línea de producción simulada mediante técnicas de aprendizaje automático clásico. Las contaminaciones corresponden a la presencia de granos de arroz sobre una superficie blanca.

Las imágenes con presencia de arroz se consideran ejemplos positivos, mientras que las imágenes sin arroz o con otros objetos corresponden a ejemplos negativos.

---

## Construcción del dataset

Inicialmente se construyó un dataset individual compuesto por:

- 15 imágenes positivas
- 15 imágenes negativas

Posteriormente, para la etapa de entrenamiento y evaluación, se utilizó un dataset grupal combinado a partir de los datos compartidos por varios estudiantes.

El dataset final utilizado para el entrenamiento se encuentra en el archivo:

dataset_grupo.csv

Este contiene:

- 118 muestras
- 16384 características por imagen
- 1 columna final correspondiente a la etiqueta

---

## Preprocesamiento

Cada imagen fue procesada mediante:

1. Conversión a escala de grises
2. Redimensionamiento a 128×128 píxeles
3. Binarización:
   - 1 → fondo blanco
   - 0 → presencia de objetos

Las imágenes binarias fueron convertidas en vectores fila para construir el dataset tabular.

---

## Modelos evaluados

Se entrenaron y compararon los siguientes modelos de clasificación clásica:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Naive Bayes

La evaluación se realizó utilizando:

- partición estratificada 80/20
- múltiples particiones con diferentes valores de `random_state`

---

## Resultados

Los resultados mostraron que:

- KNN presentó un desempeño deficiente y sesgo hacia la clase positiva
- Decision Tree obtuvo resultados intermedios
- SVM mostró un comportamiento estable y buen desempeño general
- Naive Bayes obtuvo el mejor accuracy promedio considerando múltiples particiones

Por esta razón, el modelo final seleccionado fue:

Naive Bayes

con un accuracy promedio de:

0.8708

---

## Archivos principales

| Archivo | Descripción |
|---|---|
| `crear_dataset.py` | Generación del dataset individual |
| `unir_datasets.py` | Combinación de datasets grupales |
| `entrenar_modelos.py` | Entrenamiento y evaluación de modelos |
| `dataset.csv` | Dataset individual |
| `dataset_grupo.csv` | Dataset grupal utilizado para entrenamiento |
| `modelo_final.joblib` | Modelo final exportado |
| `proyecto_IA.pdf` | Informe del proyecto |

---

## Cómo ejecutar

1. Instalar dependencias
python -m pip install opencv-python numpy pandas scikit-learn joblib

2. Generar dataset individual
python crear_dataset.py

3. Combinar datasets grupales
python unir_datasets.py

4. Entrenar modelos y exportar modelo final
python entrenar_modelos.py

Autor

Luis Enrique Pereira B. C05859. Universidad de Costa Rica. 2026.
