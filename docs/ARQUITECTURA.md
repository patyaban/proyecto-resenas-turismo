# Arquitectura del proyecto

## 1. Visión general

El proyecto implementa un sistema de PLN de punta a punta para el **Caso de uso 5:
Reseñas de turismo (hoteles/restaurantes)**, cubriendo los 7 requisitos
obligatorios del enunciado: pipeline de preprocesamiento reutilizable,
modelado de temas (LDA), análisis de sentimiento (general y por aspecto),
comparación clásico vs. neuronal, interfaz de uso, y un repositorio de código
documentado.

```
                        ┌─────────────────────────────────────────────┐
                        │                NAVEGADOR                     │
                        │   web/index.html + estilos.css + app.js      │
                        └───────────────┬─────────────────────────────┘
                                        │  HTTP (fetch)
                                        │  POST /analizar  {"texto":..., "modelo":...}
                                        ▼
                        ┌─────────────────────────────────────────────┐
                        │            API REST  (FastAPI)               │
                        │              api/app.py                      │
                        │  /analizar  /analizar_lote  /metricas  /temas │
                        └───────────────┬─────────────────────────────┘
                                        │  Predictor.predecir(texto, modelo)
                                        ▼
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
        ▼                               ▼                               ▼
┌───────────────┐             ┌───────────────────┐           ┌──────────────────┐
│ MODELO CLÁSICO │             │  MODELO NEURONAL   │           │  ANÁLISIS DE      │
│ (Naive Bayes)  │             │  (GRU bidireccional)│           │  ASPECTOS + LDA   │
│ .joblib        │             │  .keras + tokenizer │           │  léxico + LDA     │
└───────────────┘             └───────────────────┘           └──────────────────┘
        │                               │                               │
        └───────────────┬───────────────┘                               │
                        ▼                                               ▼
              sentimiento_general + confianza          aspectos (4) + tema_dominante
                        └───────────────────┬───────────────────────────┘
                                            ▼
                          estrellas_estimadas (fórmula que combina ambos)
```

## 2. Por qué DOS ramas de entrenamiento (requisito #4)

El enunciado exige comparar un enfoque **clásico** (Naive Bayes) contra uno
**neuronal** (LSTM/GRU o Transformer). Por eso el proyecto tiene DOS scripts de
entrenamiento independientes que predicen la MISMA columna
(`sentimiento_general`) sobre la MISMA partición train/test:

```
data/resenas_turismo.csv
        │
        ├──────────────────────────────┬───────────────────────────────┐
        ▼                              ▼                               ▼
src/entrenar_clasico.py     src/entrenar_neuronal.py         src/modelado_temas.py
  TF-IDF + Naive Bayes         Tokenizer + GRU bidireccional     CountVectorizer + LDA
        │                              │                               │
        ▼                              ▼                               ▼
modelo_clasico_nb.joblib      modelo_neuronal.keras            lda_modelo.joblib
metricas_clasico.json         metricas_neuronal.json           temas_lda.json
        │                              │
        └──────────────┬───────────────┘
                        ▼
              src/comparar_modelos.py
                        │
                        ▼
              modelos/comparacion.json  →  GET /metricas
```

Usar el mismo `train_test_split(..., random_state=42, stratify=y)` en ambos
scripts es clave: si cada modelo se evaluara con una partición distinta, la
comparación de F1/exactitud no sería justa.

## 3. Del sentimiento general a "estrellas + aspectos"

El enunciado, para este caso de uso, pide que la interfaz muestre
**"estrellas estimadas + aspectos"**, no solo un sentimiento de 3 clases. Esto
se resuelve en `src/analisis_aspectos.py` y se orquesta en `src/predecir.py`:

1. El modelo (clásico o neuronal) da el `sentimiento_general` + `confianza`.
2. Un léxico de polaridad + palabras clave por aspecto detecta, para cada uno
   de los 4 aspectos de negocio (comida, servicio, limpieza, precio), si se
   menciona y con qué polaridad — **sin necesidad de entrenar un modelo
   supervisado adicional por aspecto** (no tendríamos etiquetas a nivel de
   oración en un corpus real).
3. `estimar_estrellas()` combina ambas señales (sentimiento general +
   aspectos) en una fórmula explicable con una sola línea de justificación
   (ver el docstring de esa función) — importante para poder defender
   "por qué el sistema dio 4 estrellas y no 5" en la presentación en vivo.

## 4. Decisiones de diseño y su justificación

| Decisión | Por qué |
|----------|---------|
| **Un solo `preprocesamiento.py` reutilizado en 5 módulos** | Cumple el requisito #1 al pie de la letra: un pipeline reutilizable, no cinco limpiezas de texto ligeramente distintas. |
| **Naive Bayes para el modelo "clásico"** | Es el clasificador clásico que pide explícitamente el enunciado; rápido, interpretable, buen baseline para texto disperso (TF-IDF). |
| **GRU bidireccional para el modelo "neuronal"** | Cumple la opción "LSTM/GRU" del enunciado sin depender de descargar pesos preentrenados de Hugging Face (a diferencia de BETO), lo cual es más robusto en un entorno de clase/demo sin garantía de internet. |
| **LDA con scikit-learn (no gensim)** | No agrega una dependencia nueva: el proyecto ya usa scikit-learn para todo lo demás. |
| **Aspectos vía léxico + palabras clave, no ML supervisado por aspecto** | El dataset (sintético o real) no siempre trae etiquetas a nivel de oración; el enfoque léxico es transparente, no requiere entrenamiento adicional y es fácil de explicar en la defensa. |
| **Estrellas derivadas por fórmula, no por un modelo de regresión aparte** | Mantiene la trazabilidad: se puede explicar exactamente por qué el sistema asignó N estrellas a partir del sentimiento y los aspectos. |
| **`class_weight`/reproducibilidad (semillas fijas) en ambos entrenamientos** | Permite que la comparación clásico vs. neuronal sea repetible y comparable entre corridas. |
| **FastAPI sirve API + web (mismo patrón que el proyecto de referencia)** | Un solo comando (`uvicorn`) levanta todo: interfaz, API de análisis, métricas y temas. |
| **Modelos NO versionados en git** | Se regeneran ejecutando los scripts de `src/`; versionarlos infla el repo con binarios que quedan obsoletos en cuanto alguien reentrena. |

## 5. Estructura de carpetas

```
proyecto-resenas-turismo/
├── api/
│   ├── __init__.py
│   └── app.py                  API REST + servidor web
├── data/
│   └── resenas_turismo.csv     Dataset sintético (reseñas + etiquetas)
├── docs/
│   └── ARQUITECTURA.md         Este documento
├── src/
│   ├── __init__.py
│   ├── preprocesamiento.py     Pipeline reutilizable (requisito #1)
│   ├── generar_dataset.py      Genera el dataset sintético de reseñas
│   ├── modelado_temas.py       LDA (requisito #2)
│   ├── analisis_aspectos.py    Sentimiento por aspecto + fórmula de estrellas
│   ├── entrenar_clasico.py     Naive Bayes (mitad "clásica" del requisito #4)
│   ├── entrenar_neuronal.py    GRU bidireccional (mitad "neuronal" del requisito #4)
│   ├── comparar_modelos.py     Tabla comparativa clásico vs. neuronal
│   └── predecir.py             Orquesta todo: sentimiento + aspectos + estrellas + tema
├── modelos/                    Modelos y métricas serializados (generados, no versionados)
├── web/                        Interfaz web (requisito #5)
│   ├── index.html
│   ├── estilos.css
│   └── app.js
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt
```

## 6. Mapeo directo a los requisitos obligatorios del enunciado

| # | Requisito del enunciado | Dónde se cumple |
|---|---|---|
| 1 | Pipeline de preprocesamiento reutilizable | `src/preprocesamiento.py`, usado por los otros 4 módulos de `src/` |
| 2 | Modelado de temas (LDA) sobre el corpus | `src/modelado_temas.py`, expuesto en `GET /temas` |
| 3 | Análisis de sentimientos (general y por aspecto) | `src/entrenar_clasico.py` / `src/entrenar_neuronal.py` (general) + `src/analisis_aspectos.py` (por aspecto) |
| 4 | Comparación clásico vs. neuronal con métricas | `src/entrenar_clasico.py`, `src/entrenar_neuronal.py`, `src/comparar_modelos.py`, expuesto en `GET /metricas` |
| 5 | Interfaz de uso funcional | `web/` + `api/app.py` (modo "una opinión": pega una reseña → estrellas + aspectos) |
| 6 | Presentación final con demo en vivo | Fuera del código: usar la interfaz web corriendo en vivo |
| 7 | Informe breve + repositorio de código | [`INFORME.md`](../INFORME.md) (resultados reales e interpretación) + el propio repositorio |
