import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


# ==========================
# CARGAR DATASET GRUPAL
# ==========================

print("Cargando dataset grupal...")

df = pd.read_csv("dataset_grupo.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("Dataset cargado correctamente.")
print("Filas:", X.shape[0])
print("Características:", X.shape[1])

print("\nDistribución total de etiquetas:")
print(y.value_counts())


# ==========================
# MODELOS CLÁSICOS
# ==========================

modelos = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "SVM": SVC(kernel="linear"),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Naive Bayes": GaussianNB()
}


# ==========================
# EVALUACIÓN 80/20
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nDatos divididos:")
print("Train:", X_train.shape[0])
print("Test:", X_test.shape[0])

print("\nDistribución en entrenamiento:")
print(y_train.value_counts())

print("\nDistribución en prueba:")
print(y_test.value_counts())

print("\n=========================")
print("RESULTADOS CON SPLIT 80/20")
print("=========================")

for nombre, modelo in modelos.items():

    print(f"\nModelo: {nombre}")

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    print("\nReporte:")
    print(classification_report(y_test, y_pred, zero_division=0))


# ==========================
# EVALUACIÓN CON VARIOS SPLITS
# ==========================

print("\n=========================")
print("PRUEBA CON VARIOS RANDOM_STATE")
print("=========================")

random_states = [0, 1, 2, 3, 4, 5, 10, 20, 42, 100]

mejor_nombre = ""
mejor_promedio = -1
mejor_modelo_base = None

for nombre, modelo in modelos.items():

    accuracies = []

    for rs in random_states:
        X_train_rs, X_test_rs, y_train_rs, y_test_rs = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=rs,
            stratify=y
        )

        modelo.fit(X_train_rs, y_train_rs)
        y_pred_rs = modelo.predict(X_test_rs)

        acc = accuracy_score(y_test_rs, y_pred_rs)
        accuracies.append(acc)

    promedio = sum(accuracies) / len(accuracies)
    minimo = min(accuracies)
    maximo = max(accuracies)

    print(f"\nModelo: {nombre}")
    print("Accuracies:", accuracies)
    print(f"Promedio : {promedio:.4f}")
    print(f"Mínimo   : {minimo:.4f}")
    print(f"Máximo   : {maximo:.4f}")

    if promedio > mejor_promedio:
        mejor_promedio = promedio
        mejor_nombre = nombre
        mejor_modelo_base = modelo


# ==========================
# ENTRENAR MODELO FINAL
# ==========================

print("\n=========================")
print("MODELO FINAL SELECCIONADO")
print("=========================")
print("Modelo:", mejor_nombre)
print(f"Accuracy promedio: {mejor_promedio:.4f}")

# Entrenar el mejor modelo con TODOS los datos disponibles
mejor_modelo_base.fit(X, y)

nombre_archivo = "modelo_final.joblib"
joblib.dump(mejor_modelo_base, nombre_archivo)

print("\nModelo final entrenado con todo el dataset grupal.")
print("Modelo exportado como:")
print(nombre_archivo)