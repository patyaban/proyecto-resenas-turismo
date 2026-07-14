"""
Entrenamiento del modelo NEURONAL: LSTM/GRU con Keras 
Uso:
    python -m src.entrenar_neuronal
    python -m src.entrenar_neuronal --epocas 15 --dim-embedding 64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.preprocesamiento import normalizar

LONGITUD_MAXIMA = 40   
TAMANO_VOCABULARIO = 4000


def construir_modelo(dim_embedding: int, n_clases: int) -> keras.Model:
    """Arquitectura: Embedding -> GRU bidireccional -> Dropout -> Dense(softmax)."""
    modelo = keras.Sequential([
        keras.layers.Input(shape=(LONGITUD_MAXIMA,)),
        keras.layers.Embedding(
            input_dim=TAMANO_VOCABULARIO,
            output_dim=dim_embedding,
            mask_zero=True,   # ignora el padding (ceros) al procesar la secuencia
        ),
        keras.layers.Bidirectional(keras.layers.GRU(32)),
        keras.layers.Dropout(0.3),   # regularización: apaga 30% de neuronas al azar en entrenamiento
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(n_clases, activation="softmax"),
    ])
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",  # las etiquetas son enteros (0,1,2)
        metrics=["accuracy"],
    )
    return modelo


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el modelo neuronal (GRU) de sentimiento.")
    parser.add_argument("--datos", default="data/resenas_turismo.csv")
    parser.add_argument("--modelo", default="modelos/modelo_neuronal.keras")
    parser.add_argument("--tokenizer", default="modelos/tokenizer_neuronal.joblib")
    parser.add_argument("--codificador", default="modelos/codificador_etiquetas.joblib")
    parser.add_argument("--metricas", default="modelos/metricas_neuronal.json")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epocas", type=int, default=12)
    parser.add_argument("--dim-embedding", type=int, default=32)
    parser.add_argument("--objetivo", default="sentimiento_general")
    args = parser.parse_args()

    # --- 1. Cargar y normalizar datos ---------------------------------------
    df = pd.read_csv(args.datos)
    df = df.dropna(subset=["texto", args.objetivo])
    print(f"[1/6] Datos cargados: {len(df)} reseñas")

    
    # Keras (a diferencia de TfidfVectorizer) no acepta un `preprocessor`.
    textos_normalizados = df["texto"].astype(str).apply(normalizar)
    etiquetas_texto = df[args.objetivo].astype(str)

    codificador = LabelEncoder()
    y = codificador.fit_transform(etiquetas_texto)   # "negativo"->0, "neutral"->1, "positivo"->2 (alfabético)
    n_clases = len(codificador.classes_)

    # --- 2. Partición train / test (idéntica lógica que el modelo clásico) -
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        textos_normalizados, y, test_size=args.test_size, random_state=42, stratify=y,
    )
    print(f"[2/6] Partición -> train: {len(X_train_txt)}  test: {len(X_test_txt)}")

    # --- 3. Tokenización y padding -------------------------------------------
    tokenizer = Tokenizer(num_words=TAMANO_VOCABULARIO, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_txt)

    X_train_seq = pad_sequences(
        tokenizer.texts_to_sequences(X_train_txt),
        maxlen=LONGITUD_MAXIMA, padding="post", truncating="post",
    )
    X_test_seq = pad_sequences(
        tokenizer.texts_to_sequences(X_test_txt),
        maxlen=LONGITUD_MAXIMA, padding="post", truncating="post",
    )
    print("[3/6] Textos tokenizados y convertidos a secuencias de longitud fija")

    # --- 4. Construir y entrenar el modelo -----------------------------------
    modelo = construir_modelo(args.dim_embedding, n_clases)
    modelo.summary()

    parada_temprana = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True,
    )

    historial = modelo.fit(
        X_train_seq, y_train,
        validation_split=0.15,
        epochs=args.epocas,
        batch_size=32,
        callbacks=[parada_temprana],
        verbose=2,
    )
    print("[4/6] Modelo neuronal entrenado")

    # --- 5. Evaluar -----------------------------------------------------------
    probas = modelo.predict(X_test_seq)
    y_pred = probas.argmax(axis=1)

    nombres_clases = list(codificador.classes_)
    reporte = classification_report(
        y_test, y_pred, target_names=nombres_clases, output_dict=True, zero_division=0,
    )
    matriz = confusion_matrix(y_test, y_pred)

    print("\n[5/6] Evaluación sobre el conjunto de prueba:")
    print(classification_report(y_test, y_pred, target_names=nombres_clases, zero_division=0))
    print(f"Etiquetas: {nombres_clases}")
    print("Matriz de confusión (filas=real, columnas=predicho):")
    print(matriz)

    # --- 6. Guardar modelo, tokenizer y métricas ------------------------------
    Path(args.modelo).parent.mkdir(parents=True, exist_ok=True)
    modelo.save(args.modelo)
    joblib.dump(tokenizer, args.tokenizer)
    joblib.dump(codificador, args.codificador)

    metricas = {
        "modelo": "gru_bidireccional_neuronal",
        "objetivo": args.objetivo,
        "exactitud": reporte["accuracy"],
        "f1_macro": reporte["macro avg"]["f1-score"],
        "precision_macro": reporte["macro avg"]["precision"],
        "recall_macro": reporte["macro avg"]["recall"],
        "reporte_por_clase": {k: v for k, v in reporte.items() if k in nombres_clases},
        "matriz_confusion": matriz.tolist(),
        "n_entrenamiento": len(X_train_txt),
        "n_prueba": len(X_test_txt),
        "clases": nombres_clases,
        "epocas_entrenadas": len(historial.history["loss"]),
        "arquitectura": f"Embedding({args.dim_embedding}) -> BiGRU(32) -> Dropout(0.3) -> Dense(16) -> Dense({n_clases}, softmax)",
    }
    with open(args.metricas, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    print(f"\n[6/6] Modelo guardado en    : {args.modelo}")
    print(f"       Tokenizer guardado en: {args.tokenizer}")
    print(f"       Métricas guardadas en: {args.metricas}")
    print(f"       Exactitud: {metricas['exactitud']:.3f} | F1 macro: {metricas['f1_macro']:.3f}")


if __name__ == "__main__":
    main()

