"""
API REST del sistema de análisis de reseñas de turismo + servidor de la
interfaz web (Requisito obligatorio #5: "Interfaz de uso funcional").

El enunciado sugiere Gradio, Streamlit o ipywidgets para la interfaz, pero
como este proyecto sigue la MISMA arquitectura del proyecto de referencia
(FastAPI + HTML/CSS/JS servidos por el mismo proceso), usamos ese enfoque:
cumple igual el requisito ("una interfaz simple que reciba texto... y muestre
los resultados... FUNCIONE"), y además queda todo en un solo comando
(`uvicorn`), sin depender de Colab.

Expone:
    GET  /                -> interfaz web (HTML)
    GET  /salud           -> estado del servicio y de los modelos cargados
    GET  /metricas        -> métricas comparadas (clásico vs. neuronal)
    GET  /temas           -> palabras clave de los temas descubiertos por LDA
    POST /analizar        -> {"texto": "...", "modelo": "clasico"|"neuronal"}
                              -> sentimiento + confianza + aspectos + estrellas + tema
    POST /analizar_lote   -> {"textos": ["...", "..."]}  -> lista de resultados

Ejecutar:
    uvicorn api.app:app --reload
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.predecir import obtener_predictor

RAIZ = Path(__file__).resolve().parent.parent
DIR_WEB = RAIZ / "web"
RUTA_METRICA_CLASICO = RAIZ / "modelos" / "metricas_clasico.json"
RUTA_METRICA_NEURONAL = RAIZ / "modelos" / "metricas_neuronal.json"
RUTA_COMPARACION = RAIZ / "modelos" / "comparacion.json"
RUTA_TEMAS_LDA = RAIZ / "modelos" / "temas_lda.json"

app = FastAPI(
    title="API de Análisis de Reseñas de Turismo",
    description=(
        "Analiza reseñas de hoteles y restaurantes en español: sentimiento "
        "general, sentimiento por aspecto (comida, servicio, limpieza, "
        "precio), estrellas estimadas y tema dominante (LDA)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class EntradaTexto(BaseModel):
    texto: str = Field(..., min_length=1, examples=[
        "La comida estuvo deliciosa y el servicio fue muy atento, aunque los precios son un poco altos."
    ])
    modelo: str = Field(default="clasico", pattern="^(clasico|neuronal)$",
                         description="Qué modelo usar para el sentimiento general: 'clasico' (Naive Bayes) o 'neuronal' (GRU).")


class EntradaLote(BaseModel):
    textos: list[str] = Field(..., min_length=1)
    modelo: str = Field(default="clasico", pattern="^(clasico|neuronal)$")


def _predictor():
    try:
        return obtener_predictor()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


# --- Endpoints de sistema ----------------------------------------------------

@app.get("/salud", tags=["sistema"])
def salud():
    predictor = None
    try:
        predictor = _predictor()
    except HTTPException:
        pass
    return {
        "estado": "ok",
        "modelo_clasico_cargado": predictor is not None,
        "modelo_neuronal_cargado": predictor.modelo_neuronal_disponible if predictor else False,
        "modelo_lda_cargado": predictor.lda_bundle is not None if predictor else False,
    }


@app.get("/metricas", tags=["sistema"])
def metricas():
    """Métricas comparadas del modelo clásico vs. el neuronal (requisito #4)."""
    if RUTA_COMPARACION.exists():
        return json.loads(RUTA_COMPARACION.read_text(encoding="utf-8"))

    # Si aún no se corrió src/comparar_modelos.py, devolvemos lo que exista
    # por separado en vez de fallar.
    resultado = {}
    if RUTA_METRICA_CLASICO.exists():
        resultado["clasico"] = json.loads(RUTA_METRICA_CLASICO.read_text(encoding="utf-8"))
    if RUTA_METRICA_NEURONAL.exists():
        resultado["neuronal"] = json.loads(RUTA_METRICA_NEURONAL.read_text(encoding="utf-8"))
    if not resultado:
        raise HTTPException(status_code=404, detail="Aún no hay métricas. Entrena los modelos primero.")
    return resultado


@app.get("/temas", tags=["sistema"])
def temas():
    """Palabras clave de los temas descubiertos por LDA (requisito #2)."""
    if not RUTA_TEMAS_LDA.exists():
        raise HTTPException(status_code=404, detail="Aún no hay temas. Ejecuta: python -m src.modelado_temas")
    return json.loads(RUTA_TEMAS_LDA.read_text(encoding="utf-8"))


# --- Endpoints de inferencia --------------------------------------------------

@app.post("/analizar", tags=["inferencia"])
def analizar(entrada: EntradaTexto):
    """Analiza una única reseña: sentimiento + aspectos + estrellas + tema."""
    return _predictor().predecir(entrada.texto, modelo=entrada.modelo)


@app.post("/analizar_lote", tags=["inferencia"])
def analizar_lote(entrada: EntradaLote):
    """Analiza una lista de reseñas (modo panel/lote sugerido en el enunciado)."""
    predictor = _predictor()
    resultados = [predictor.predecir(t, modelo=entrada.modelo) for t in entrada.textos]

    # Resumen agregado para el "modo panel" que pide la sección 5 del enunciado.
    from collections import Counter
    conteo_sentimiento = Counter(r["sentimiento_general"] for r in resultados)
    promedio_estrellas = round(sum(r["estrellas_estimadas"] for r in resultados) / len(resultados), 2)

    return {
        "resultados": resultados,
        "resumen": {
            "n_resenas": len(resultados),
            "distribucion_sentimiento": dict(conteo_sentimiento),
            "promedio_estrellas": promedio_estrellas,
        },
    }


# --- Interfaz web --------------------------------------------------------------

@app.get("/", include_in_schema=False)
def raiz():
    return FileResponse(DIR_WEB / "index.html")


app.mount("/static", StaticFiles(directory=DIR_WEB), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=True)
