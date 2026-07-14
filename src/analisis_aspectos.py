from __future__ import annotations

import re

ASPECTOS = ["comida", "servicio", "limpieza", "precio"]

# Palabras clave que indican que se está hablando de cada aspecto.
PALABRAS_CLAVE_ASPECTO = {
    "comida": [
        "comida", "platillo", "platillos", "plato", "platos", "desayuno",
        "cena", "almuerzo", "menu", "menú", "gastronomia", "gastronomía",
        "sabor", "cocina", "postre", "bebida", "bebidas",
    ],
    "servicio": [
        "servicio", "personal", "atencion", "atención", "staff", "mesero",
        "meseros", "mesera", "recepcion", "recepción", "recepcionista",
        "trato", "amabilidad",
    ],
    "limpieza": [
        "limpieza", "limpio", "limpia", "sucio", "sucia", "higiene",
        "habitacion", "habitación", "cuarto", "bano", "baño", "aseo", "orden",
    ],
    "precio": [
        "precio", "precios", "costo", "costos", "tarifa", "tarifas", "caro",
        "cara", "barato", "barata", "economico", "económico", "pagar",
        "vale", "valor",
    ],
}

LEXICO_POSITIVO = {
    "excelente", "increible", "increíble", "delicioso", "deliciosa",
    "exquisito", "exquisita", "sabroso", "sabrosa", "impecable", "impoluto",
    "limpio", "limpia", "reluciente", "atento", "atenta", "amable",
    "profesional", "rapido", "rápido", "rapida", "rápida", "accesible",
    "justo", "razonable", "recomendable", "recomiendo", "cordial",
    "encanto", "encantó", "genial", "fantastico", "fantástico", "fantastica",
    "fantástica", "maravilloso", "maravillosa",
    "buenisimo", "buenísimo", "buenisima", "buenísima",
    "perfecto", "perfecta", "cuidada", "cuidado",
    "bueno", "buena", "buenos", "buenas", "agradable", "comodo", "cómodo",
    "comoda", "cómoda", "rico", "rica", "ricos", "ricas",
    "riquisimo", "riquísimo", "riquisima", "riquísima",
}

LEXICO_NEGATIVO = {
    "pesimo", "pésimo", "pesima", "pésima", "horrible", "malo", "mala",
    "insipido", "insípida", "insipida", "sucio", "sucia", "descuidado",
    "descuidada", "lento", "lenta", "descortes", "descortés", "deficiente",
    "excesivo", "excesiva", "caro", "cara", "sobrevalorado", "sobrevalorada",
    "decepcionante", "decepcionado", "decepcionada", "terrible", "grasoso",
    "grasosa", "frio", "fría", "fria", "olor", "demorado", "demorada",
    "indiferente", "incomodo", "incómodo", "incomoda", "incómoda",
}

# Negaciones que invierten la polaridad de la palabra que las sigue.
NEGACIONES = {"no", "nunca", "ni", "tampoco", "nada"}

_RE_ORACIONES = re.compile(r"[.!?,;]+")
_RE_PALABRA = re.compile(r"[a-záéíóúñü]+")


def _dividir_oraciones(texto: str) -> list[str]:
    """Divide un texto en cláusulas cortas usando puntuación (incluida la coma)
    como separador.

    Se incluye la coma a propósito: en reseñas es muy común escribir varios
    aspectos en una sola oración separados por comas ("la comida estuvo
    deliciosa, el servicio fue lento y los precios son altos"). Sin dividir
    por comas, la polaridad de UN aspecto terminaría "contaminada" por las
    palabras de los otros aspectos mencionados en la misma oración larga.
    """
    partes = _RE_ORACIONES.split(texto.lower())
    return [p.strip() for p in partes if p.strip()]


def _polaridad_oracion(oracion: str) -> int:
    """Calcula un puntaje de polaridad para una oración (+1, 0, -1 por palabra).

    Si una palabra del léxico está precedida por una negación dentro de una
    ventana de 2 posiciones (p. ej. "no fue bueno"), su polaridad se invierte.
    """
    palabras = _RE_PALABRA.findall(oracion)
    puntaje = 0
    for i, palabra in enumerate(palabras):
        polaridad = 0
        if palabra in LEXICO_POSITIVO:
            polaridad = 1
        elif palabra in LEXICO_NEGATIVO:
            polaridad = -1
        else:
            continue

        ventana = palabras[max(0, i - 2):i]
        if any(v in NEGACIONES for v in ventana):
            polaridad *= -1

        puntaje += polaridad
    return puntaje


def analizar_aspectos(texto: str) -> dict:
    """Devuelve la polaridad detectada para cada uno de los 4 aspectos.

    Returns:
        dict con claves "comida", "servicio", "limpieza", "precio", cada una
        con valor "positivo" / "negativo" / "neutral" / "no_mencionado".
    """
    oraciones = _dividir_oraciones(texto)
    resultado = {}

    for aspecto in ASPECTOS:
        claves = PALABRAS_CLAVE_ASPECTO[aspecto]
        oraciones_del_aspecto = [
            o for o in oraciones if any(clave in o for clave in claves)
        ]

        if not oraciones_del_aspecto:
            resultado[aspecto] = "no_mencionado"
            continue

        puntaje_total = sum(_polaridad_oracion(o) for o in oraciones_del_aspecto)

        if puntaje_total > 0:
            resultado[aspecto] = "positivo"
        elif puntaje_total < 0:
            resultado[aspecto] = "negativo"
        else:
            resultado[aspecto] = "neutral"

    return resultado


def estimar_estrellas(sentimiento_general: str, confianza: float, aspectos: dict) -> int:
    """Combina el sentimiento general (del clasificador) con los aspectos
    detectados (del léxico) en una estimación de estrellas de 1 a 5.

    Fórmula:
        base = 3  (punto neutro)
        + 2 * confianza  si el sentimiento general es positivo
        - 2 * confianza  si el sentimiento general es negativo
        + ajuste por aspectos: cada aspecto positivo suma 0.25, cada negativo resta 0.25
    El resultado se redondea y se recorta al rango [1, 5].
    """
    base = 3.0
    if sentimiento_general == "positivo":
        base += 2 * confianza
    elif sentimiento_general == "negativo":
        base -= 2 * confianza

    ajuste_aspectos = sum(
        0.25 if v == "positivo" else -0.25 if v == "negativo" else 0.0
        for v in aspectos.values()
    )
    base += ajuste_aspectos

    estrellas = round(base)
    return max(1, min(5, estrellas))


if __name__ == "__main__":
    # Demostración ejecutable: python -m src.analisis_aspectos
    ejemplos = [
        "La comida estuvo deliciosa y el servicio fue muy atento. "
        "Sin embargo, la limpieza de la habitación no fue buena.",
        "Los precios son excesivos y el personal fue descortés.",
    ]
    for e in ejemplos:
        print(f"TEXTO: {e}")
        print(f"ASPECTOS: {analizar_aspectos(e)}")
        print("-" * 60)
