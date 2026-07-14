"""
Generador de dataset sintético de reseñas de turismo (hoteles y restaurantes) 

Uso:
    python -m src.generar_dataset                  # genera data/resenas_turismo.csv
    python -m src.generar_dataset --n 1200          # genera 1200 reseñas
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

# --- Vocabulario base -------------------------------------------------------

ASPECTOS = ["comida", "servicio", "limpieza", "precio"]

ESTABLECIMIENTOS = {
    "hotel": [
        "el Hotel Andino", "el Hotel Illimani", "el Hostal Sol Naciente",
        "el Hotel Plaza Mayor", "el Hotel Mirador del Valle", "el Hostal Kantuta",
        "el Hotel Los Pinos", "el Hotel Real Audiencia",
    ],
    "restaurante": [
        "el Restaurante La Paceña", "el Restaurante Sabor Andino",
        "la Parrilla del Valle", "el Restaurante Rincón Cochabambino",
        "el Restaurante Fogón Criollo", "el Café Illimani",
        "el Restaurante Los Yungas", "la Cantina Altiplano",
    ],
}

# adjetivo/expresión por aspecto y por sentimiento
ADJ_ASPECTO = {
    "comida": {
        "positivo": ["deliciosa", "exquisita", "muy sabrosa", "excelente",
                     "increíble", "de primera calidad", "recién preparada y sabrosa"],
        "negativo": ["insípida", "de mala calidad", "fría cuando debía estar caliente",
                     "decepcionante", "poco fresca", "muy grasosa"],
        "neutral": ["aceptable", "normal", "correcta", "estándar", "sin nada especial"],
    },
    "servicio": {
        "positivo": ["muy atento", "rápido y amable", "profesional", "excelente",
                     "muy cordial", "atento en todo momento"],
        "negativo": ["lento", "descortés", "pésimo", "deficiente", "indiferente",
                     "muy demorado"],
        "neutral": ["aceptable", "correcto", "normal", "estándar"],
    },
    "limpieza": {
        "positivo": ["impecable", "muy limpio", "reluciente", "impoluto",
                     "cuidada hasta el último detalle"],
        "negativo": ["descuidada", "con mal olor", "deficiente", "sucia",
                     "por debajo de lo esperado"],
        "neutral": ["aceptable", "normal", "adecuada"],
    },
    "precio": {
        "positivo": ["muy accesible", "justo", "razonable", "económico para lo que ofrece",
                     "una excelente relación precio-calidad"],
        "negativo": ["excesivo", "muy caro", "sobrevalorado", "no vale lo que cuesta",
                     "más alto de lo que justifica la experiencia"],
        "neutral": ["promedio", "similar al de la competencia", "razonable dentro del mercado"],
    },
}

# plantillas de frase por aspecto (usan {adj})
FRASES_ASPECTO = {
    "comida": [
        "la comida estuvo {adj}",
        "los platillos fueron {adj}",
        "la comida es {adj}",
        "el desayuno fue {adj}",
    ],
    "servicio": [
        "el servicio fue {adj}",
        "el personal fue {adj}",
        "nos atendieron de forma {adj}",
        "la atención del staff fue {adj}",
    ],
    "limpieza": [
        "la limpieza del lugar es {adj}",
        "las habitaciones estaban {adj}",
        "todo se veía {adj}",
        "el aseo general fue {adj}",
    ],
    "precio": [
        "los precios son {adj}",
        "la relación precio-calidad es {adj}",
        "el costo nos pareció {adj}",
        "las tarifas resultaron {adj}",
    ],
}

CONECTORES = ["Además,", "También,", "Por otro lado,", "En cuanto al resto,", "Asimismo,"]

# Puntaje numérico por sentimiento, usado para derivar sentimiento general y estrellas.
PUNTAJE_SENTIMIENTO = {"positivo": 1, "neutral": 0, "negativo": -1}


def _capitalizar(texto: str) -> str:
    return texto[0].upper() + texto[1:] if texto else texto


def generar_resena(rng: random.Random) -> dict:
    """Genera una única reseña sintética con sus etiquetas derivadas."""
    tipo = rng.choice(["hotel", "restaurante"])
    nombre = rng.choice(ESTABLECIMIENTOS[tipo])

    n_mencionados = rng.randint(2, 4)
    aspectos_mencionados = rng.sample(ASPECTOS, n_mencionados)

    etiquetas_aspecto = {a: "no_mencionado" for a in ASPECTOS}
    clausulas = []
    puntaje_total = 0

    for aspecto in aspectos_mencionados:
        sentimiento = rng.choice(["positivo", "negativo", "neutral"])
        adj = rng.choice(ADJ_ASPECTO[aspecto][sentimiento])
        clausula = rng.choice(FRASES_ASPECTO[aspecto]).format(adj=adj)
        clausulas.append(clausula)
        etiquetas_aspecto[aspecto] = sentimiento
        puntaje_total += PUNTAJE_SENTIMIENTO[sentimiento]

   
    oraciones = [f"En {nombre}, {clausulas[0]}"]
    for clausula in clausulas[1:]:
        oraciones.append(f"{rng.choice(CONECTORES)} {clausula}")
    texto = ". ".join(oraciones) + "."
    texto = _capitalizar(texto)

    promedio = puntaje_total / n_mencionados  # rango [-1, 1]

    if promedio > 0.25:
        sentimiento_general = "positivo"
    elif promedio < -0.25:
        sentimiento_general = "negativo"
    else:
        sentimiento_general = "neutral"

    # Mapeo de promedio [-1, 1] a estrellas [1, 5]: 3 es el punto neutro.
    estrellas = round(3 + promedio * 2)
    estrellas = max(1, min(5, estrellas))

    fila = {
        "texto": texto,
        "tipo": tipo,
        "sentimiento_general": sentimiento_general,
        "estrellas": estrellas,
    }
    for aspecto in ASPECTOS:
        fila[f"aspecto_{aspecto}"] = etiquetas_aspecto[aspecto]
    return fila


def generar(n: int, semilla: int = 42) -> list[dict]:
    """Genera `n` reseñas únicas (sin duplicados exactos de texto)."""
    rng = random.Random(semilla)
    filas: list[dict] = []
    vistos: set[str] = set()

    intentos = 0
    max_intentos = n * 50
    while len(filas) < n and intentos < max_intentos:
        intentos += 1
        fila = generar_resena(rng)
        if fila["texto"] in vistos:
            continue
        vistos.add(fila["texto"])
        filas.append(fila)

    rng.shuffle(filas)
    return filas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera el dataset sintético de reseñas de turismo (hoteles y restaurantes).")
    parser.add_argument("--n", type=int, default=1000, help="Número total de reseñas a generar.")
    parser.add_argument("--salida", type=str, default="data/resenas_turismo.csv",
                         help="Ruta del CSV de salida.")
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args()

    filas = generar(args.n, args.semilla)

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)

    columnas = ["texto", "tipo", "sentimiento_general", "estrellas",
                "aspecto_comida", "aspecto_servicio", "aspecto_limpieza", "aspecto_precio"]

    with salida.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(filas)

    print(f"[OK] {len(filas)} reseñas escritas en {salida}")

    from collections import Counter
    print("\nDistribución por sentimiento general:")
    for clase, cant in sorted(Counter(f["sentimiento_general"] for f in filas).items()):
        print(f"     {clase:>9}: {cant}")

    print("\nDistribución por estrellas:")
    for estrella, cant in sorted(Counter(f["estrellas"] for f in filas).items()):
        print(f"     {estrella}★: {cant}")

    print("\nDistribución por tipo de establecimiento:")
    for tipo, cant in sorted(Counter(f["tipo"] for f in filas).items()):
        print(f"     {tipo:>11}: {cant}")


if __name__ == "__main__":
    main()
