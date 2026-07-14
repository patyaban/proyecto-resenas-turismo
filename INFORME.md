# Informe Breve — Analizador de Reseñas de Turismo

**Proyecto final · Módulo de Procesamiento de Lenguaje Natural**
**Maestría en Ciencia de Datos e Inteligencia Artificial Aplicada · Universidad Católica Boliviana "San Pablo"**
**Caso de uso 5 del enunciado: Reseñas de turismo (hoteles/restaurantes)**

> Todos los números de este informe fueron obtenidos ejecutando los scripts del
> repositorio (`src/entrenar_clasico.py`, `src/entrenar_neuronal.py`,
> `src/modelado_temas.py`, `src/comparar_modelos.py`) sobre el dataset final de
> 1000 reseñas. No hay cifras estimadas ni de ejemplo: son el resultado real
> de la última corrida.

---

## 1. El problema y el caso de uso elegido

Elegimos el **Caso de uso 5** del enunciado: *Reseñas de turismo
(hoteles/restaurantes)*. El problema que resuelve: una reseña de texto libre
no le dice a un negocio *qué* específicamente estuvo bien o mal — un
sentimiento general ("positivo") no le indica al gerente de un hotel si debe
revisar la cocina, capacitar al personal, revisar la limpieza o ajustar
tarifas. Este sistema recibe una reseña en español y entrega:

- El **sentimiento general** (positivo/negativo/neutral), con dos enfoques distintos y comparables.
- El **sentimiento por aspecto** (comida, servicio, limpieza, precio).
- Una **estimación de estrellas** (1 a 5) derivada de ambas señales.
- El **tema dominante** del texto, según un modelo de descubrimiento de temas (LDA) entrenado sobre todo el corpus.

Está dirigido a un administrador de hotel/restaurante que quiere una lectura
rápida y accionable de sus reseñas, sin tener que leerlas todas manualmente.

## 2. Los datos: origen, tamaño y cómo se prepararon

**Origen:** el enunciado sugiere reseñas reales de TripAdvisor en español. Por
restricciones de tiempo y acceso a un corpus etiquetado, se optó por un
**dataset sintético generado por reglas** (`src/generar_dataset.py`):
combina plantillas de oración con vocabulario de 4 aspectos de negocio
(comida, servicio, limpieza, precio), 16 nombres de establecimientos (hoteles
y restaurantes) y una semilla fija (`semilla=42`) para que sea reproducible.

**Tamaño:** 1000 reseñas, partidas en 800 de entrenamiento / 200 de prueba
(80/20, estratificado por `sentimiento_general`).

**Cómo se etiquetó:** a diferencia de un etiquetado manual, aquí las
etiquetas se **derivan matemáticamente**: cada reseña combina entre 2 y 4
aspectos mencionados al azar, cada uno con su propia polaridad; el
`sentimiento_general` y las `estrellas` se calculan a partir de esas
polaridades (ver fórmula en `src/generar_dataset.py` y
`src/analisis_aspectos.py::estimar_estrellas`). Esto garantiza que el
dataset sea internamente coherente, aunque **no reemplaza la variabilidad y
ambigüedad del lenguaje real** (ironía, errores ortográficos, opiniones
mixtas complejas) — ver limitaciones en la sección 6.

**Distribución obtenida** (real, del archivo final `data/resenas_turismo.csv`):

| Sentimiento general | Reseñas |
|---|---|
| neutral | 371 |
| positivo | 322 |
| negativo | 307 |

| Estrellas | Reseñas |
|---|---|
| 1★ | 58 |
| 2★ | 309 |
| 3★ | 258 |
| 4★ | 319 |
| 5★ | 56 |

| Tipo de establecimiento | Reseñas |
|---|---|
| restaurante | 522 |
| hotel | 478 |

Nótese el desbalance en los extremos de la escala de estrellas (1★ y 5★
tienen bastantes menos casos que 2★-4★): esto es consistente con la lógica
del generador — una reseña solo llega a 1 o 5 estrellas si TODOS los
aspectos que menciona coinciden en polaridad extrema, algo estadísticamente
menos frecuente que una mezcla. Se discute como desafío en la sección 6.

## 3. El pipeline, paso a paso

```
texto crudo
   │
   ▼
Preprocesamiento reutilizable (src/preprocesamiento.py)
   minúsculas → quitar URLs/menciones/números → quitar puntuación
   → normalizar espacios → quitar stopwords (SIN quitar negaciones)
   │
   ├──────────────────────────────┬────────────────────────────┐
   ▼                              ▼                             ▼
Representación TF-IDF      Representación Embeddings      Representación CountVectorizer
(clásico)                  (neuronal)                      (LDA)
   │                              │                             │
   ▼                              ▼                             ▼
Naive Bayes                GRU bidireccional               LDA (4 temas)
   │                              │                             │
   └──────────────┬───────────────┘                             │
                  ▼                                              │
        sentimiento_general + confianza                          │
                  │                                               │
                  ▼                                               ▼
     Léxico de polaridad (src/analisis_aspectos.py)      tema_dominante
                  │
                  ▼
    aspectos (comida/servicio/limpieza/precio) + estrellas estimadas
```

La misma función `normalizar()` se reutiliza en los 5 módulos que procesan
texto (Requisito obligatorio #1), evitando que entrenamiento e inferencia
usen limpiezas distintas.

## 4. Comparación de modelos: clásico (Naive Bayes) vs. neuronal (GRU)

Ambos modelos se entrenaron y evaluaron sobre la **misma partición**
train/test (800/200, estratificada, `random_state=42`), prediciendo la
misma columna (`sentimiento_general`), para que la comparación sea justa.

### Tabla de métricas (macro-promedio)

| Métrica | Clásico (Naive Bayes) | Neuronal (GRU bidireccional) |
|---|---|---|
| Exactitud | 67.0% | 89.5% |
| Precisión macro | 69.9% | 89.7% |
| Recall macro | 66.9% | 89.8% |
| F1 macro | 67.8% | 89.7% |

### Métricas por clase

**Clásico (Naive Bayes):**

| Clase | Precisión | Recall | F1 | Soporte (n en test) |
|---|---|---|---|---|
| negativo | 79.6% | 62.9% | 70.3% | 62 |
| neutral | 55.4% | 68.9% | 61.4% | 74 |
| positivo | 74.6% | 68.8% | 71.5% | 64 |

**Neuronal (GRU bidireccional):**

| Clase | Precisión | Recall | F1 | Soporte (n en test) |
|---|---|---|---|---|
| negativo | 90.8% | 95.2% | 92.9% | 62 |
| neutral | 86.3% | 85.1% | 85.7% | 74 |
| positivo | 91.9% | 89.1% | 90.5% | 64 |

### Matrices de confusión (filas = real, columnas = predicho; orden: negativo, neutral, positivo)

**Clásico:**
```
              neg   neu   pos
real neg    [  39,   23,    0 ]
real neu    [   8,   51,   15 ]
real pos    [   2,   18,   44 ]
```

**Neuronal:**
```
              neg   neu   pos
real neg    [  59,    3,    0 ]
real neu    [   6,   63,    5 ]
real pos    [   0,    7,   57 ]
```

### Interpretación

- El modelo **neuronal supera claramente al clásico** en las 4 métricas
  (aprox. +22 puntos porcentuales de exactitud). La ventaja más marcada está
  en **negativo** (recall de 62.9% → 95.2%): el Naive Bayes confunde 23 de
  62 reseñas negativas con "neutral" (ver primera fila de su matriz), algo
  que el GRU prácticamente resuelve (solo 3 confusiones).
- La clase que más cuesta a **ambos** modelos es **"neutral"**: tiene la
  precisión más baja en los dos casos (55.4% y 86.3% respectivamente). Esto
  es esperable: "neutral" es, por construcción, la clase intermedia — más
  fácil de confundir con positivo o negativo que estos dos entre sí.
- La arquitectura neuronal usada es `Embedding(32) → GRU bidireccional(32) →
  Dropout(0.3) → Dense(16) → Dense(3, softmax)`, entrenada 12 épocas con
  `EarlyStopping` monitoreando la pérdida de validación.
- **Hipótesis de por qué gana el GRU en este dataset:** Naive Bayes trata
  cada palabra de forma independiente (bolsa de palabras), mientras que el
  GRU procesa la reseña en orden y puede aprender relaciones como la
  posición de una negación respecto al adjetivo que modifica. Como el
  dataset sintético combina varias cláusulas de aspecto con conectores
  ("además", "sin embargo"...), el orden de las palabras aporta información
  real que el GRU aprovecha y Naive Bayes no.

## 5. Modelado de temas (LDA)

Se ajustó un modelo LDA con 4 temas (`n_components=4`, elegido para que
coincida con los 4 aspectos de negocio) sobre el corpus completo, usando
`CountVectorizer` (no TF-IDF, ver justificación en `docs/ARQUITECTURA.md`).

**Palabras clave descubiertas por tema (reales, sin editar):**

| Tema | Palabras clave |
|---|---|
| 0 | restaurante, estaban, habitaciones, rincón, cochabambino, personal, comida, aseo, general, servicio |
| 1 | hotel, aseo, general, andino, comida, costo, pareció, valle, forma, atendieron |
| 2 | razonable, lugar, limpieza, comida, forma, atendieron, resultaron, tarifas, dentro, mercado |
| 3 | calidad, precio, relación, hostal, comida, excelente, habitaciones, estaban, sol, naciente |

**Interpretación honesta:** los temas **no separan limpiamente** los 4
aspectos de negocio como se esperaba en el diseño inicial — la palabra
"comida" aparece en los 4 temas, y varios temas mezclan nombres propios de
establecimientos ("cochabambino", "andino", "sol naciente") con vocabulario
de aspecto. Esto sugiere que, en un corpus generado por plantillas con pocos
nombres de establecimiento repetidos muchas veces, LDA termina agrupando
documentos parcialmente por **qué lugar se menciona**, no solo por **de qué
aspecto se habla**. Es una observación legítima para la defensa (ver sección
6) y un punto de mejora concreto para la sección 7.

## 6. Desafíos encontrados y decisiones tomadas

- **Negación:** se excluyeron explícitamente "no", "nunca", "nada",
  "tampoco" de las stopwords (`src/preprocesamiento.py`). Sin esta decisión,
  frases como "el servicio no fue bueno" perderían la negación y se leerían
  como positivas.
- **Aspectos en una misma oración:** las reseñas reales (y las sintéticas)
  suelen mencionar varios aspectos separados por comas en una sola oración
  larga. La primera versión de `analisis_aspectos.py` calculaba la
  polaridad sobre la oración completa, lo que "contaminaba" un aspecto con
  el sentimiento de otro mencionado en la misma oración. Se corrigió
  dividiendo también por comas, no solo por puntos, acotando la ventana de
  contexto a la cláusula relevante.
- **Desbalance en los extremos de la escala de estrellas:** como se detalla
  en la sección 2, 1★ y 5★ tienen menos ejemplos por construcción. No se
  aplicó una corrección adicional porque el objetivo de clasificación
  comparado (`sentimiento_general`, 3 clases) está razonablemente balanceado
  (307/371/322); queda como mejora pendiente si se decide clasificar
  directamente por estrellas.
- **Datos sintéticos vs. reales:** el vocabulario es más limpio y
  consistente que el lenguaje real (sin errores ortográficos, sarcasmo,
  jerga local), lo cual probablemente **infla las métricas** respecto a lo
  que se obtendría con reseñas reales de TripAdvisor.
- **LDA no separó los temas como se esperaba** (ver sección 5) — un
  hallazgo honesto más que un error de código: con pocos establecimientos
  repetidos muchas veces, sus nombres se vuelven señales fuertes que
  compiten con el vocabulario de aspecto.
- **Cobertura incompleta del léxico de aspectos (concordancia de género):**
  durante las pruebas manuales de la interfaz se detectó que la frase "la
  comida es buenísima" marcaba el aspecto "comida" como neutral, porque el
  léxico solo incluía la forma masculina ("buenísimo") y no la femenina
  ("buenísima"), que es la que concuerda con "comida". Se corrigió agregando
  las formas de género faltantes (`src/analisis_aspectos.py`). Es un
  recordatorio concreto de que un léxico escrito a mano nunca queda
  "completo" a la primera — necesita iteración con casos reales.

## 7. Conclusiones, limitaciones y posibles mejoras

**Conclusiones:**
- El sistema cumple el flujo completo pedido por el enunciado: limpieza →
  representación (TF-IDF y embeddings) → temas (LDA) → sentimiento general y
  por aspecto → comparación clásico vs. neuronal con métricas → interfaz de
  uso.
- En este dataset, el enfoque neuronal (GRU) superó claramente al clásico
  (Naive Bayes) en las 4 métricas evaluadas.

**Limitaciones:**
- El dataset es sintético; no captura toda la variabilidad del lenguaje
  real (ironía, errores, jerga).
- Los aspectos se detectan con un léxico de polaridad manual, no con un
  modelo supervisado — es transparente e interpretable, pero limitado al
  vocabulario que se incluyó a mano.
- Los temas de LDA no se alinearon limpiamente con los 4 aspectos de
  negocio esperados.

**Mejoras futuras:**
- Reemplazar el dataset sintético por reseñas reales de TripAdvisor en
  español y comparar la caída de las métricas.
- Entrenar un modelo supervisado de aspectos si se consigue un corpus con
  anotaciones a nivel de oración.
- Probar BETO (Transformer preentrenado en español) en vez de GRU si se
  cuenta con acceso a internet durante el entrenamiento (código de reemplazo
  ya documentado en `src/entrenar_neuronal.py`).
- Aumentar la variedad de nombres de establecimientos y vocabulario para que
  LDA tenga menos señal "de nombre propio" compitiendo con el vocabulario de
  aspecto.

