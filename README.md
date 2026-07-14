# 🏨 Analizador de Reseñas de Turismo (Hoteles y Restaurantes)

Proyecto final del módulo de **Procesamiento de Lenguaje Natural** — Maestría en
Ciencia de Datos e Inteligencia Artificial Aplicada, Universidad Católica Boliviana "San Pablo".

**Caso de uso elegido (#5 del enunciado):** *Reseñas de turismo (hoteles/restaurantes)* —
sentimiento por aspecto (comida, servicio, limpieza, precio). La interfaz recibe
una reseña y devuelve **estrellas estimadas + aspectos**.

```
Preprocesamiento  →  TF-IDF/Embeddings  →  LDA (temas)  →  Naive Bayes / GRU  →  Aspectos + Estrellas
```

> Diagrama completo y justificación de cada decisión en [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md).

---

## ✅ Cobertura de los requisitos obligatorios del enunciado

| # | Requisito | Script(s) / documento |
|---|---|---|
| 1 | Pipeline de preprocesamiento reutilizable | `src/preprocesamiento.py` |
| 2 | Modelado de temas (LDA) | `src/modelado_temas.py` |
| 3 | Análisis de sentimientos (general + por aspecto) | `src/entrenar_clasico.py`, `src/entrenar_neuronal.py`, `src/analisis_aspectos.py` |
| 4 | Comparación clásico vs. neuronal con métricas | `src/entrenar_clasico.py` (Naive Bayes) + `src/entrenar_neuronal.py` (GRU) + `src/comparar_modelos.py` |
| 5 | Interfaz de uso funcional | `web/` + `api/app.py` |

---

## 🚀 Puesta en marcha

Desde la carpeta `proyecto-resenas-turismo/`:

### 1. Crear el entorno e instalar dependencias

```bash
python3.11 -m venv venv
source venv/bin/activate          # en Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> `tensorflow` (para el modelo neuronal) es la dependencia más pesada. Si tu
> equipo tiene poca RAM/espacio, puedes omitirla: el resto del sistema
> (modelo clásico, LDA, aspectos, API e interfaz) funciona igual sin ella —
> simplemente no podrás entrenar/usar el modelo "neuronal" en la interfaz.

### 2. Generar el dataset sintético

```bash
python -m src.generar_dataset            # crea data/resenas_turismo.csv (1000 reseñas)
```

### 3. Entrenar los modelos

```bash
python -m src.entrenar_clasico            # Naive Bayes -> modelos/modelo_clasico_nb.joblib
python -m src.entrenar_neuronal           # GRU         -> modelos/modelo_neuronal.keras
python -m src.modelado_temas              # LDA         -> modelos/lda_modelo.joblib
python -m src.comparar_modelos            # tabla comparativa -> modelos/comparacion.json
```

### 4. Levantar la aplicación web + API

```bash
uvicorn api.app:app --reload
```

Abre 👉 **http://127.0.0.1:8000** — pega una reseña y obtén sentimiento,
estrellas y aspectos. La documentación interactiva está en
**http://127.0.0.1:8000/docs**.

---

## 🗂️ Estructura

```
proyecto-resenas-turismo/
├── api/app.py                  API REST + servidor de la interfaz web
├── data/resenas_turismo.csv    Dataset sintético (generado)
├── docs/ARQUITECTURA.md        Diagramas y justificación de diseño
├── src/
│   ├── preprocesamiento.py       Pipeline de limpieza reutilizable
│   ├── generar_dataset.py        Genera el dataset sintético
│   ├── modelado_temas.py         LDA sobre el corpus
│   ├── analisis_aspectos.py      Sentimiento por aspecto + fórmula de estrellas
│   ├── entrenar_clasico.py       Naive Bayes
│   ├── entrenar_neuronal.py      GRU bidireccional
│   ├── comparar_modelos.py       Compara métricas clásico vs. neuronal
│   └── predecir.py               Orquesta todo para la inferencia
├── modelos/                     Modelos y métricas (generados, no versionados)
├── web/                          Interfaz (HTML + CSS + JS)
├── requirements.txt
└── README.md
```

---

## 🧪 Probar sin la interfaz web

**CLI:**
```bash
python -m src.predecir "La comida estuvo deliciosa pero el servicio fue lento"
python -m src.predecir --modelo neuronal "..."
python -m src.predecir            # modo interactivo
```

**curl:**
```bash
curl -X POST http://127.0.0.1:8000/analizar \
     -H "Content-Type: application/json" \
     -d '{"texto":"La limpieza fue impecable pero los precios muy altos","modelo":"clasico"}'
```

Respuesta esperada (aproximada):
```json
{
  "sentimiento_general": "neutral",
  "confianza": 0.55,
  "estrellas_estimadas": 3,
  "aspectos": {
    "comida": "no_mencionado",
    "servicio": "no_mencionado",
    "limpieza": "positivo",
    "precio": "negativo"
  }
}
```

---

## 📊 Sobre el dataset

- **1000 reseñas** sintéticas de hoteles y restaurantes, generadas combinando
  plantillas + vocabulario por aspecto (reproducible con semilla fija, ver
  `src/generar_dataset.py --semilla`).
- Cada reseña combina de 2 a 4 aspectos mencionados al azar, cada uno con su
  propio sentimiento; el `sentimiento_general` y las `estrellas` se derivan
  matemáticamente de esos aspectos (dataset internamente consistente).
- **Desbalance esperado:** hay menos reseñas de 1★/5★ que de 2★-4★ (una
  reseña necesita que TODOS sus aspectos mencionados coincidan en polaridad
  extrema para llegar a los bordes de la escala) — es un buen punto para
  discutir desbalance de clases en la presentación (`class_weight`,
  estratificación, etc. — ver `docs/ARQUITECTURA.md`).
- Para una práctica más realista, sustituye `data/resenas_turismo.csv` por
  reseñas reales de TripAdvisor en español y compara cómo cambian las
  métricas de ambos modelos.

---

## 🔌 Endpoints de la API

| Método | Ruta               | Descripción                                          |
|--------|--------------------|-------------------------------------------------------|
| GET    | `/`                | Interfaz web                                          |
| GET    | `/salud`           | Estado del servicio y de los modelos cargados         |
| GET    | `/metricas`        | Comparación clásico vs. neuronal (requisito #4)       |
| GET    | `/temas`           | Palabras clave de los temas descubiertos por LDA      |
| POST   | `/analizar`        | Analiza una reseña → sentimiento + aspectos + estrellas |
| POST   | `/analizar_lote`   | Analiza una lista de reseñas + resumen agregado       |
| GET    | `/docs`            | Documentación interactiva (Swagger UI)                |

---

## 🎓 Recomendaciones

1. Sustituir el dataset sintético por reseñas reales de TripAdvisor en
   español y comparar la caída de exactitud.
2. Ampliar `entrenar_neuronal.py` para usar BETO (Transformer preentrenado en
   español) en vez de GRU si el entorno de defensa tiene acceso a internet.   
3. Entrenar un modelo de aspectos supervisado (en vez de léxico) si se
   consigue un corpus con anotaciones a nivel de oración.
4. Agregar un modo "lote" en la interfaz web que suba un CSV y muestre el
   resumen agregado que ya devuelve `POST /analizar_lote`.
