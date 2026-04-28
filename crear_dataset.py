import os
import cv2
import numpy as np
import pandas as pd

# Carpeta donde está este script
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

POS_PATH = os.path.join(BASE_PATH, "dataset", "positivas")
NEG_PATH = os.path.join(BASE_PATH, "dataset", "negativas")

GRIS_PATH = os.path.join(BASE_PATH, "procesadas", "grises")
BIN_PATH = os.path.join(BASE_PATH, "procesadas", "binarias")

os.makedirs(GRIS_PATH, exist_ok=True)
os.makedirs(BIN_PATH, exist_ok=True)

data = []

print("Ejecutando script desde:")
print(BASE_PATH)
print("Guardando grises en:")
print(GRIS_PATH)
print("Guardando binarias en:")
print(BIN_PATH)


def procesar_imagenes(carpeta, etiqueta):
    print(f"\nProcesando: {carpeta}")

    if not os.path.exists(carpeta):
        print("ERROR: No existe esta carpeta.")
        return

    for nombre_archivo in os.listdir(carpeta):
        ruta_imagen = os.path.join(carpeta, nombre_archivo)

        if not os.path.isfile(ruta_imagen):
            continue

        imagen = cv2.imread(ruta_imagen)

        if imagen is None:
            print(f"No se pudo leer: {ruta_imagen}")
            continue

        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        redimensionada = cv2.resize(gris, (128, 128))

        _, binaria = cv2.threshold(redimensionada, 127, 1, cv2.THRESH_BINARY)
        binaria_vis = binaria * 255

        # Guardar como PNG para evitar problemas con JPEG
        nombre_sin_ext = os.path.splitext(nombre_archivo)[0]

        ruta_gris = os.path.join(GRIS_PATH, nombre_sin_ext + "_gris.png")
        ruta_binaria = os.path.join(BIN_PATH, nombre_sin_ext + "_binaria.png")

        ok_gris = cv2.imwrite(ruta_gris, redimensionada)
        ok_binaria = cv2.imwrite(ruta_binaria, binaria_vis)

        print(f"{nombre_archivo} -> gris: {ok_gris}, binaria: {ok_binaria}")

        vector = binaria.flatten()
        vector_con_etiqueta = np.append(vector, etiqueta)
        data.append(vector_con_etiqueta)


procesar_imagenes(POS_PATH, 1)
procesar_imagenes(NEG_PATH, 0)

df = pd.DataFrame(data)

csv_path = os.path.join(BASE_PATH, "dataset.csv")
df.to_csv(csv_path, index=False)

print("\nCSV generado correctamente.")
print("Archivo CSV:", csv_path)
print("Filas:", df.shape[0])
print("Columnas:", df.shape[1])