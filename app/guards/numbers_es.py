from __future__ import annotations

import re
import unicodedata
from decimal import Decimal

_UNITS = {
    "cero": 0,
    "uno": 1,
    "una": 1,
    "un": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "dieciseis": 16,
    "diecisiete": 17,
    "dieciocho": 18,
    "diecinueve": 19,
    "veinte": 20,
    "veintiuno": 21,
    "veintidos": 22,
    "veintitres": 23,
    "veinticuatro": 24,
    "veinticinco": 25,
    "veintiseis": 26,
    "veintisiete": 27,
    "veintiocho": 28,
    "veintinueve": 29,
}
_TENS = {
    "treinta": 30,
    "cuarenta": 40,
    "cincuenta": 50,
    "sesenta": 60,
    "setenta": 70,
    "ochenta": 80,
    "noventa": 90,
}
_COMPACT_TENS = {
    f"{tens}{unit}": tens_value + unit_value
    for tens, tens_value in _TENS.items()
    for unit, unit_value in _UNITS.items()
    if 1 <= unit_value <= 9 and unit not in {"un", "una"}
}
_COMPACT_TENS.update(
    {
        f"{tens}i{unit}": tens_value + unit_value
        for tens, tens_value in _TENS.items()
        for unit, unit_value in _UNITS.items()
        if 1 <= unit_value <= 9 and unit not in {"un", "una"}
    }
)
_UNITS.update(_COMPACT_TENS)
_HUNDREDS = {
    "cien": 100,
    "ciento": 100,
    "doscientos": 200,
    "doscientas": 200,
    "trescientos": 300,
    "trescientas": 300,
    "cuatrocientos": 400,
    "cuatrocientas": 400,
    "quinientos": 500,
    "quinientas": 500,
    "seiscientos": 600,
    "seiscientas": 600,
    "setecientos": 700,
    "setecientas": 700,
    "ochocientos": 800,
    "ochocientas": 800,
    "novecientos": 900,
    "novecientas": 900,
}
_MILLIONS = frozenset({"millon", "millones"})
_NUMBER_WORDS = frozenset({*_UNITS, *_TENS, *_HUNDREDS, *_MILLIONS, "mil", "y"})
_QUANTITY_UNITS = frozenset(
    {
        "cuota",
        "cuotas",
        "pago",
        "pagos",
        "dia",
        "dias",
        "hora",
        "horas",
        "mes",
        "meses",
        "peso",
        "pesos",
        "porcentaje",
    }
)


def _fold(text: str) -> list[str]:
    value = "".join(
        char
        for char in unicodedata.normalize("NFKD", text).casefold()
        if not unicodedata.combining(char)
    )
    return re.findall(r"[a-z]+", value)


def _parse(words: list[str]) -> int | None:
    total = 0
    current = 0
    meaningful = False
    for word in words:
        if word == "y":
            continue
        meaningful = True
        if word in _UNITS:
            current += _UNITS[word]
        elif word in _TENS:
            current += _TENS[word]
        elif word in _HUNDREDS:
            current += _HUNDREDS[word]
        elif word == "mil":
            total += max(current, 1) * 1000
            current = 0
        else:  # callers only pass number words, so the remaining word is a million scale
            total = (total + max(current, 1)) * 1_000_000
            current = 0
    return total + current if meaningful else None


def numbers_in_words(text: str) -> tuple[Decimal, ...]:
    tokens = _fold(text)
    results: list[Decimal] = []
    index = 0
    while index < len(tokens):
        if tokens[index] not in _NUMBER_WORDS or tokens[index] == "y":
            index += 1
            continue
        end = index
        while end < len(tokens) and tokens[end] in _NUMBER_WORDS:
            end += 1
        words = tokens[index:end]
        while words and words[-1] == "y":
            words.pop()
            end -= 1
        parsed = _parse(words)
        following = tokens[end] if end < len(tokens) else None
        # "un millón" is a scaled amount, never an article.
        scaled = any(word in _MILLIONS or word == "mil" for word in words)
        if parsed is not None and (parsed != 1 or scaled or following in _QUANTITY_UNITS):
            results.append(Decimal(parsed))
        index = max(end, index + 1)
    return tuple(results)
