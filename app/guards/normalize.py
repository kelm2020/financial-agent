from __future__ import annotations

import re
import unicodedata

_INVISIBLE = re.compile("[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_SPACE = re.compile(r"\s+")

# Detection-only subset of the UTS #39 confusable skeleton relevant to Spanish/English
# injection phrases. The visible/model text is never replaced by this mapping.
_CONFUSABLES = str.maketrans(
    {
        "\u0430": "a",
        "\u0391": "A",
        "\u0410": "A",
        "\u0435": "e",
        "\u0415": "E",
        "\u0395": "E",
        "\u0456": "i",
        "\u0399": "I",
        "\u0406": "I",
        "\u0458": "j",
        "\u0408": "J",
        "\u03bf": "o",
        "\u043e": "o",
        "\u039f": "O",
        "\u041e": "O",
        "\u0440": "p",
        "\u0420": "P",
        "\u03c1": "p",
        "\u0441": "c",
        "\u0421": "C",
        "\u03f2": "c",
        "\u0455": "s",
        "\u0405": "S",
        "\u0445": "x",
        "\u0425": "X",
        "\u0443": "y",
        "\u0423": "Y",
    }
)


def normalize_visible(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _INVISIBLE.sub("", normalized)
    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) not in {"Cc", "Cf"} or character in "\n\t"
    )
    return _SPACE.sub(" ", normalized).strip()


def detection_skeleton(text: str) -> str:
    visible = normalize_visible(text).translate(_CONFUSABLES)
    decomposed = unicodedata.normalize("NFKD", visible)
    unaccented = "".join(char for char in decomposed if not unicodedata.combining(char))
    return unaccented.casefold()
