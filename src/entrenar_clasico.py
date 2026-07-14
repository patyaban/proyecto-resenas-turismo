from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.preprocesamiento import normalizar


def construir_pipeline() -> Pipeline:
    return Pipeline(steps=[
        ("tfidf", TfidfVectorizer(
            preprocessor=normalizar,
            ngram_range=(1, 2),   # unigramas + bigramas: captura "no fue" / "muy bueno"
            min_df=2,
        )),
        ("clasificador", MultinomialNB(
            alpha=0.5,   # suavizado de Laplace/Lidstone: evita probabilidad 0 para
                          # palabras del vocabulario de test que no aparecieron en
                          # una clase durante el entrenamiento
        )),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el modelo clásico (Naive Bayes).")
    parser.add_argument("--datos", default="data/resenas_turismo.csv")
    parser.add_argument("--modelo", default="modelos/modelo_clasico_nb.joblib")
    parser.add_argument("--metricas", default="modelos/metricas_clasico.json")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--objetivo", default="sentimiento_general",
                         help="Columna objetivo a predecir (por defecto: sentimiento_general).")
    args = parser.parse_args()

    # --- 1. Cargar datos ----------------------------------------------------
    df = pd.read_csv(args.datos)
    df = df.dropna(subset=["texto", args.objetivo])
    print(f"[1/5] Datos cargados: {len(df)} reseñas")
    print(df[args.objetivo].value_counts().to_string())

    X = df["texto"].astype(str)
    y = df[args.objetivo].astype(str)

    # --- 2. Partición train / test ------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y,
    )
    print(f"\n[2/5] Partición -> train: {len(X_train)}  test: {len(X_test)}")

    # --- 3. Entrenar ---------------------------------------------------------
    pipeline = construir_pipeline()
    pipeline.fit(X_train, y_train)
    print("[3/5] Modelo Naive Bayes entrenado")

    # --- 4. Evaluar ------------------------------------------------------------
    y_pred = pipeline.predict(X_test)
    reporte = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    matriz = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))

    print("\n[4/5] Evaluación sobre el conjunto de prueba:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Etiquetas: {sorted(y.unique())}")
    print("Matriz de confusión (filas=real, columnas=predicho):")
    print(matriz)

    # --- 5. Guardar modelo y métricas --------------------------------------
    Path(args.modelo).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.modelo)

    metricas = {
        "modelo": "naive_bayes_clasico",
        "objetivo": args.objetivo,
        "exactitud": reporte["accuracy"],
        "f1_macro": reporte["macro avg"]["f1-score"],
        "precision_macro": reporte["macro avg"]["precision"],
        "recall_macro": reporte["macro avg"]["recall"],
        "reporte_por_clase": {k: v for k, v in reporte.items() if k in sorted(y.unique())},
        "matriz_confusion": matriz.tolist(),
        "n_entrenamiento": len(X_train),
        "n_prueba": len(X_test),
        "clases": sorted(y.unique()),
    }
    with open(args.metricas, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    print(f"\n[5/5] Modelo guardado en   : {args.modelo}")
    print(f"      Métricas guardadas en: {args.metricas}")
    print(f"      Exactitud: {metricas['exactitud']:.3f} | F1 macro: {metricas['f1_macro']:.3f}")


if __name__ == "__main__":
    main()
