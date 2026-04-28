# Detección de contaminaciones en imágenes mediante aprendizaje automático clásico

## Descripción

Este proyecto tiene como objetivo detectar la presencia de granos de arroz en imágenes de una superficie blanca, simulando una línea de producción. Se utiliza procesamiento de imágenes y aprendizaje automático clásico.

## Dataset

El conjunto de datos está compuesto por 30 imágenes:

* 15 positivas (con presencia de arroz)
* 15 negativas (sin arroz)

Las imágenes se encuentran organizadas en la carpeta `database/`, separadas en clases.

## Preprocesamiento

Las imágenes fueron procesadas mediante:

* Conversión a escala de grises
* Redimensionamiento a 128×128 píxeles
* Binarización (fondo = 1, objeto = 0)

Las imágenes procesadas pueden encontrarse en la carpeta `procesadas/`.

## Generación del dataset

El script `crear_dataset.py` permite generar el archivo `dataset.csv`, donde:

* Cada fila representa una imagen
* Cada columna representa un píxel (16384 en total)
* La última columna corresponde a la etiqueta (1 o 0)

## Archivos incluidos

* `crear_dataset.py`: script principal
* `dataset.csv`: conjunto de datos generado
* `database/`: imágenes originales
* `procesadas/`: imágenes procesadas
* `Informe_Proyecto1.pdf`: informe del proyecto

## Cómo ejecutar

1. Instalar dependencias:

```
pip install opencv-python numpy pandas
```

2. Ejecutar:

```
python crear_dataset.py
```

## Autor

[Tu nombre]
