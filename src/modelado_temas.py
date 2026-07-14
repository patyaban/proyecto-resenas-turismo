"""
Modelado de temas con LDA 

Uso:
    python -m src.modelado_temas                     # ajusta LDA con 4 temas
    python -m src.modelado_temas --n-temas 6 --n-palabras 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from src.preprocesamiento import normalizar


def construir_vectorizador() -> CountVectorizer:
    """LDA trabaja sobre conteos de palabras (no TF-IDF): usa la frecuencia
    cruda de cada término, porque el modelo generativo de LDA asume que los
    documentos se generan mezclando temas que a su vez generan palabras según
    su frecuencia — TF-IDF ya "aplana" esa frecuencia y distorsiona el
    supuesto del modelo."""
    return CountVectorizer(
        preprocessor=normalizar,
        max_df=0.90,   # ignora palabras presentes en más del 90% de los documentos (demasiado genéricas)
        min_df=3,       # ignora palabras rarísimas (ruido, errores tipográficos)
    )


def entrenar_lda(textos: list[str], n_temas: int, semilla: int = 42):
    vectorizador = construir_vectorizador()
    matriz = vectorizador.fit_transform(textos)

    lda = LatentDirichletAllocation(
        n_components=n_temas,
        random_state=semilla,
        learning_method="batch",   # más estable y reproducible que "online" para datasets pequeños/medianos
        max_iter=25,
    )
    lda.fit(matriz)
    return lda, vectorizador, matriz


def extraer_temas(lda: LatentDirichletAllocation, vectorizador: CountVectorizer,
                   n_palabras: int = 10) -> list[dict]:
    """Para cada tema, devuelve las `n_palabras` con mayor peso.

    `lda.components_` es una matriz (n_temas x n_vocabulario) donde cada celda
    es el peso de una palabra dentro de un tema. Ordenar esa fila de mayor a
    menor y tomar las primeras N nos da las palabras más representativas.
    """
    vocabulario = vectorizador.get_feature_names_out()
    temas = []
    for indice_tema, pesos in enumerate(lda.components_):
        top_indices = pesos.argsort()[::-1][:n_palabras]
        palabras_top = [vocabulario[i] for i in top_indices]
        temas.append({
            "tema": indice_tema,
            "palabras_clave": palabras_top,
        })
    return temas


def asignar_tema_dominante(lda: LatentDirichletAllocation, matriz) -> list[int]:
    """Para cada documento, devuelve el índice del tema con mayor probabilidad
    (`transform` da, por documento, la distribución de probabilidad sobre los
    temas; tomamos el argmax)."""
    distribucion = lda.transform(matriz)
    return distribucion.argmax(axis=1).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ajusta un modelo LDA sobre el corpus de reseñas.")
    parser.add_argument("--datos", default="data/resenas_turismo.csv")
    parser.add_argument("--n-temas", type=int, default=4,
                         help="Número de temas a descubrir (sugerido: igual al número de aspectos de negocio).")
    parser.add_argument("--n-palabras", type=int, default=10,
                         help="Palabras clave a mostrar por tema.")
    parser.add_argument("--salida", default="modelos/temas_lda.json")
    args = parser.parse_args()

    df = pd.read_csv(args.datos)
    df = df.dropna(subset=["texto"])
    textos = df["texto"].astype(str).tolist()

    print(f"[1/3] Ajustando LDA con {args.n_temas} temas sobre {len(textos)} reseñas...")
    lda, vectorizador, matriz = entrenar_lda(textos, args.n_temas)

    print("[2/3] Extrayendo palabras clave por tema...")
    temas = extraer_temas(lda, vectorizador, args.n_palabras)
    for t in temas:
        print(f"  Tema {t['tema']}: {', '.join(t['palabras_clave'])}")

    tema_dominante = asignar_tema_dominante(lda, matriz)
    df_salida = df.copy()
    df_salida["tema_dominante"] = tema_dominante

    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    with open(args.salida, "w", encoding="utf-8") as f:
        json.dump({"n_temas": args.n_temas, "temas": temas}, f, ensure_ascii=False, indent=2)

    ruta_csv_temas = Path(args.salida).parent / "resenas_con_tema.csv"
    df_salida.to_csv(ruta_csv_temas, index=False)

    # Guardamos también el vectorizador + el modelo LDA ya ajustados (no solo
    # las palabras clave en JSON) para poder calcular el tema dominante de una
    # reseña NUEVA en el momento de la inferencia, sin tener que reentrenar.
    ruta_modelo_lda = Path(args.salida).parent / "lda_modelo.joblib"
    joblib.dump({"vectorizador": vectorizador, "lda": lda, "temas": temas}, ruta_modelo_lda)

    print(f"\n[3/3] Temas guardados en    : {args.salida}")
    print(f"       Reseñas + tema en    : {ruta_csv_temas}")
    print(f"       Modelo LDA guardado : {ruta_modelo_lda}")


if __name__ == "__main__":
    main()
