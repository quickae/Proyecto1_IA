import os
import pandas as pd

CARPETA_DATASETS = "datasets_grupo"
OUTPUT = "dataset_grupo.csv"

dataframes = []

for archivo in os.listdir(CARPETA_DATASETS):
    if archivo.endswith(".csv"):
        ruta = os.path.join(CARPETA_DATASETS, archivo)
        print(f"\nLeyendo: {ruta}")

        df = pd.read_csv(ruta)

        print("Filas originales:", df.shape[0])
        print("Columnas originales:", df.shape[1])

        # Si aparece una columna extra tipo índice, eliminarla
        if df.shape[1] == 16386:
            print("Detectada columna extra. Eliminando primera columna.")
            df = df.iloc[:, 1:]

        # Verificar formato correcto
        if df.shape[1] != 16385:
            print(f"ERROR: {archivo} tiene {df.shape[1]} columnas. Se omite.")
            continue

        # Forzar nombres iguales de columnas para todos los CSV
        df.columns = list(range(16385))

        dataframes.append(df)

if len(dataframes) == 0:
    print("No se encontró ningún dataset válido.")
else:
    dataset_grupo = pd.concat(dataframes, ignore_index=True)

    dataset_grupo.to_csv(OUTPUT, index=False)

    print("\n==========================")
    print("Dataset combinado generado")
    print("==========================")
    print("Archivo:", OUTPUT)
    print("Filas totales:", dataset_grupo.shape[0])
    print("Columnas:", dataset_grupo.shape[1])

    print("\nDistribución de etiquetas:")
    print(dataset_grupo.iloc[:, -1].value_counts())