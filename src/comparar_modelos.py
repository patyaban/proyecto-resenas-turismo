from __future__ import annotations

import argparse
import json
from pathlib import Path


def cargar_metricas(ruta: str) -> dict | None:
    p = Path(ruta)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def imprimir_tabla(metricas_clasico: dict, metricas_neuronal: dict) -> None:
    encabezado = f"{'Métrica':<20}{'Clásico (NB)':>18}{'Neuronal (GRU)':>18}"
    print(encabezado)
    print("-" * len(encabezado))
    filas = [
        ("Exactitud", "exactitud"),
        ("F1 macro", "f1_macro"),
        ("Precisión macro", "precision_macro"),
        ("Recall macro", "recall_macro"),
    ]
    for etiqueta, clave in filas:
        v_clasico = metricas_clasico.get(clave, float("nan"))
        v_neuronal = metricas_neuronal.get(clave, float("nan"))
        print(f"{etiqueta:<20}{v_clasico:>18.3f}{v_neuronal:>18.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara el modelo clásico vs. el neuronal.")
    parser.add_argument("--clasico", default="modelos/metricas_clasico.json")
    parser.add_argument("--neuronal", default="modelos/metricas_neuronal.json")
    parser.add_argument("--salida", default="modelos/comparacion.json")
    args = parser.parse_args()

    metricas_clasico = cargar_metricas(args.clasico)
    metricas_neuronal = cargar_metricas(args.neuronal)

    if metricas_clasico is None:
        print(f"[!] No se encontró {args.clasico}. Ejecuta primero: python -m src.entrenar_clasico")
        return
    if metricas_neuronal is None:
        print(f"[!] No se encontró {args.neuronal}. Ejecuta primero: python -m src.entrenar_neuronal")
        return

    print("=" * 60)
    print("COMPARACIÓN: MODELO CLÁSICO (Naive Bayes) vs. NEURONAL (GRU)")
    print("=" * 60)
    imprimir_tabla(metricas_clasico, metricas_neuronal)

    ganador = "clasico" if metricas_clasico["f1_macro"] >= metricas_neuronal["f1_macro"] else "neuronal"
    diferencia = abs(metricas_clasico["f1_macro"] - metricas_neuronal["f1_macro"])
    print(f"\nModelo con mayor F1 macro: {ganador}  (diferencia: {diferencia:.3f})")

    resumen = {
        "clasico": {
            "modelo": metricas_clasico.get("modelo"),
            "exactitud": metricas_clasico.get("exactitud"),
            "f1_macro": metricas_clasico.get("f1_macro"),
            "precision_macro": metricas_clasico.get("precision_macro"),
            "recall_macro": metricas_clasico.get("recall_macro"),
            "reporte_por_clase": metricas_clasico.get("reporte_por_clase"),
            "matriz_confusion": metricas_clasico.get("matriz_confusion"),
            "clases": metricas_clasico.get("clases"),
        },
        "neuronal": {
            "modelo": metricas_neuronal.get("modelo"),
            "exactitud": metricas_neuronal.get("exactitud"),
            "f1_macro": metricas_neuronal.get("f1_macro"),
            "precision_macro": metricas_neuronal.get("precision_macro"),
            "recall_macro": metricas_neuronal.get("recall_macro"),
            "reporte_por_clase": metricas_neuronal.get("reporte_por_clase"),
            "matriz_confusion": metricas_neuronal.get("matriz_confusion"),
            "clases": metricas_neuronal.get("clases"),
        },
        "mejor_modelo_por_f1_macro": ganador,
    }

    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    with open(args.salida, "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    print(f"\nComparación guardada en: {args.salida}")
    print("(Este JSON es el que consume el endpoint GET /metricas de la API)")


if __name__ == "__main__":
    main()
