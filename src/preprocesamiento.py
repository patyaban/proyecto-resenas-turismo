from __future__ import annotations

import re
import unicodedata

# Stopwords en español (lista reducida y curada). Se excluyen a propósito las
# negaciones porque invierten el sentimiento y el análisis por aspecto depende
# de detectarlas correctamente (p. ej. "la limpieza no fue buena").
STOPWORDS_ES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al",
    "a", "ante", "con", "en", "para", "por", "sin", "sobre", "tras", "y", "o",
    "u", "e", "que", "se", "su", "sus", "mi", "mis", "me", "lo", "le", "les",
    "es", "son", "fue", "ser", "está", "están", "este", "esta", "estos",
    "estas", "ese", "esa", "eso", "muy", "más", "menos", "ya", "también",
    "pero", "como", "cuando", "porque", "si", "yo", "tú", "él", "ella", "nos",
    # Conectores discursivos frecuentes en reseñas ("por otro lado", "además",
    # "en cuanto al resto"...). No aportan significado de aspecto/sentimiento
    # y sin ellos los temas descubiertos por LDA quedan mucho más limpios.
    "además", "asimismo", "resto", "cuanto", "otro", "otros", "lado",
    "todo", "todos",
}

_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_MENCION = re.compile(r"@\w+")
_RE_NUMERO = re.compile(r"\d+")
_RE_PUNTUACION = re.compile(r"[^\w\sáéíóúñü]", flags=re.UNICODE)
_RE_ESPACIOS = re.compile(r"\s+")


def quitar_tildes(texto: str) -> str:
    """Reemplaza vocales acentuadas por su versión sin tilde (á -> a)."""
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalizar(
    texto: str,
    quitar_stopwords: bool = True,
    remover_tildes: bool = False,
) -> str:
    """Aplica toda la cadena de limpieza y devuelve el texto normalizado.

    Args:
        texto: cadena de entrada (reseña cruda de hotel/restaurante).
        quitar_stopwords: si True, elimina palabras vacías (menos negaciones).
        remover_tildes: si True, quita las tildes al final del proceso.

    Returns:
        Texto limpio, en minúsculas y con espacios normalizados.
    """
    if not isinstance(texto, str):
        texto = str(texto)

    texto = texto.lower()
    texto = _RE_URL.sub(" ", texto)
    texto = _RE_MENCION.sub(" ", texto)
    texto = _RE_NUMERO.sub(" ", texto)
    texto = _RE_PUNTUACION.sub(" ", texto)
    texto = _RE_ESPACIOS.sub(" ", texto).strip()

    if quitar_stopwords:
        palabras = [p for p in texto.split() if p not in STOPWORDS_ES]
        texto = " ".join(palabras)

    if remover_tildes:
        texto = quitar_tildes(texto)

    return texto


if __name__ == "__main__":
    
    ejemplos = [
        "¡La habitación estaba IMPECABLE!! Volveríamos sin duda a www.hotelandino.com",
        "El servicio no fue bueno, tardaron 40 minutos en atendernos.",
        "Los precios son razonables para lo que ofrece el lugar.",
    ]
    for e in ejemplos:
        print(f"ORIGINAL   : {e}")
        print(f"NORMALIZADO: {normalizar(e)}")
        print("-" * 60)
