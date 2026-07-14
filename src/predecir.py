"""
Uso CLI:
    python -m src.predecir "La comida estuvo deliciosa pero el servicio fue lento"
    python -m src.predecir --modelo neuronal "..."
    python -m src.predecir            # modo interactivo
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from src.analisis_aspectos import analizar_aspectos, estimar_estrellas
from src.preprocesamiento import normalizar

RUTA_MODELO_CLASICO = "modelos/modelo_clasico_nb.joblib"
RUTA_MODELO_NEURONAL = "modelos/modelo_neuronal.keras"
RUTA_TOKENIZER_NEURONAL = "modelos/tokenizer_neuronal.joblib"
RUTA_CODIFICADOR_NEURONAL = "modelos/codificador_etiquetas.joblib"
RUTA_MODELO_LDA = "modelos/lda_modelo.joblib"
LONGITUD_MAXIMA_NEURONAL = 40


class Predictor:
   
    def __init__(
        self,
        ruta_clasico: str = RUTA_MODELO_CLASICO,
        ruta_neuronal: str = RUTA_MODELO_NEURONAL,
        ruta_tokenizer: str = RUTA_TOKENIZER_NEURONAL,
        ruta_codificador: str = RUTA_CODIFICADOR_NEURONAL,
        ruta_lda: str = RUTA_MODELO_LDA,
    ):
        ruta = Path(ruta_clasico)
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo clásico en '{ruta}'. "
                f"Ejecuta primero: python -m src.entrenar_clasico"
            )
        self.pipeline_clasico = joblib.load(ruta)
        self.clases_clasico = list(self.pipeline_clasico.named_steps["clasificador"].classes_)

       
        self.modelo_neuronal = None
        self.tokenizer_neuronal = None
        self.codificador_neuronal = None
        self._cargar_modelo_neuronal(ruta_neuronal, ruta_tokenizer, ruta_codificador)

       
        self.lda_bundle = None
        ruta_lda_path = Path(ruta_lda)
        if ruta_lda_path.exists():
            self.lda_bundle = joblib.load(ruta_lda_path)

    def _cargar_modelo_neuronal(self, ruta_neuronal, ruta_tokenizer, ruta_codificador) -> None:
        if not (Path(ruta_neuronal).exists() and Path(ruta_tokenizer).exists()
                and Path(ruta_codificador).exists()):
            return
        try:
            from tensorflow import keras
            self.modelo_neuronal = keras.models.load_model(ruta_neuronal)
            self.tokenizer_neuronal = joblib.load(ruta_tokenizer)
            self.codificador_neuronal = joblib.load(ruta_codificador)
        except ImportError:
            
            self.modelo_neuronal = None

    @property
    def modelo_neuronal_disponible(self) -> bool:
        return self.modelo_neuronal is not None

    # --- Sentimiento ---------------------------------------------------------

    def _predecir_clasico(self, texto: str) -> tuple[str, dict]:
        etiqueta = self.pipeline_clasico.predict([texto])[0]
        probas = self.pipeline_clasico.predict_proba([texto])[0]
        distribucion = {c: round(float(p), 4) for c, p in zip(self.clases_clasico, probas)}
        return etiqueta, distribucion

    def _predecir_neuronal(self, texto: str) -> tuple[str, dict]:
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        texto_normalizado = normalizar(texto)
        secuencia = self.tokenizer_neuronal.texts_to_sequences([texto_normalizado])
        secuencia = pad_sequences(secuencia, maxlen=LONGITUD_MAXIMA_NEURONAL,
                                   padding="post", truncating="post")
        probas = self.modelo_neuronal.predict(secuencia, verbose=0)[0]
        clases = list(self.codificador_neuronal.classes_)
        distribucion = {c: round(float(p), 4) for c, p in zip(clases, probas)}
        etiqueta = clases[int(np.argmax(probas))]
        return etiqueta, distribucion

    # --- Tema dominante (LDA) -------------------------------------------------

    def _tema_dominante(self, texto: str) -> dict | None:
        if self.lda_bundle is None:
            return None
        vectorizador = self.lda_bundle["vectorizador"]
        lda = self.lda_bundle["lda"]
        temas = self.lda_bundle["temas"]

        matriz = vectorizador.transform([texto])
        distribucion = lda.transform(matriz)[0]
        indice_tema = int(distribucion.argmax())
        return {
            "tema": indice_tema,
            "probabilidad": round(float(distribucion[indice_tema]), 4),
            "palabras_clave": temas[indice_tema]["palabras_clave"],
        }

    # --- API pública -----------------------------------------------------------

    def predecir(self, texto: str, modelo: str = "clasico") -> dict:
        """Devuelve sentimiento, aspectos, estrellas y (si existe) tema dominante.

        Args:
            texto: la reseña cruda.
            modelo: "clasico" (Naive Bayes, siempre disponible) o
                    "neuronal" (GRU, si fue entrenado).
        """
        usar_neuronal = modelo == "neuronal" and self.modelo_neuronal_disponible
        if usar_neuronal:
            etiqueta, probabilidades = self._predecir_neuronal(texto)
        else:
            etiqueta, probabilidades = self._predecir_clasico(texto)

        confianza = max(probabilidades.values())
        aspectos = analizar_aspectos(texto)
        estrellas = estimar_estrellas(etiqueta, confianza, aspectos)

        resultado = {
            "texto": texto,
            "modelo_usado": "neuronal" if usar_neuronal else "clasico",
            "modelo_neuronal_disponible": self.modelo_neuronal_disponible,
            "sentimiento_general": etiqueta,
            "confianza": round(float(confianza), 4),
            "probabilidades": probabilidades,
            "aspectos": aspectos,
            "estrellas_estimadas": estrellas,
        }

        tema = self._tema_dominante(texto)
        if tema is not None:
            resultado["tema_dominante"] = tema

        return resultado


@lru_cache(maxsize=1)
def obtener_predictor() -> Predictor:
   
    return Predictor()


def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Predice sentimiento + aspectos + estrellas de una reseña.")
    parser.add_argument("texto", nargs="*", help="Texto de la reseña (opcional; si se omite, modo interactivo).")
    parser.add_argument("--modelo", choices=["clasico", "neuronal"], default="clasico")
    args = parser.parse_args()

    predictor = obtener_predictor()
    if not predictor.modelo_neuronal_disponible and args.modelo == "neuronal":
        print("[!] El modelo neuronal no está entrenado todavía; usando el modelo clásico.")

    if args.texto:
        _imprimir(predictor.predecir(" ".join(args.texto), modelo=args.modelo))
        return

    print("Modo interactivo. Escribe una reseña (o 'salir'):")
    while True:
        try:
            texto = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if texto.lower() in {"salir", "exit", "quit", ""}:
            break
        _imprimir(predictor.predecir(texto, modelo=args.modelo))


def _imprimir(resultado: dict) -> None:
    estrellas = "★" * resultado["estrellas_estimadas"] + "☆" * (5 - resultado["estrellas_estimadas"])
    print(f"  Sentimiento: {resultado['sentimiento_general'].upper()} "
          f"(confianza {resultado['confianza']:.0%}, modelo {resultado['modelo_usado']})")
    print(f"  Estrellas  : {estrellas}")
    print("  Aspectos   :")
    for aspecto, valor in resultado["aspectos"].items():
        print(f"      {aspecto:>9}: {valor}")


if __name__ == "__main__":
    _cli()
