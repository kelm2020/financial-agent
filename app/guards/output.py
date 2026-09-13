from __future__ import annotations

import re
import unicodedata
from datetime import date
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict

from app.guards.codes import GuardFlag
from app.guards.normalize import detection_skeleton
from app.guards.numbers_es import numbers_in_words

COMPLIANCE_LEXICON_PATH = Path(__file__).with_name("compliance_lexicon.yaml")

_NUMBER = re.compile(r"(?<![\w])(?:\$\s*)?\d[\d.\s]*(?:,\d+)?\s*%?")
_PHONE = re.compile(r"(?:\+\d[\d\s\-().]{8,}\d|\b0800[\s-]*\d{3}[\s-]*\d{4}\b|\b\d{10,13}\b)")
_URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
# Scheme-less domains ("pagosya.com.ar", "bit.ly/x") are contacts too.
_BARE_DOMAIN = re.compile(
    r"(?i)(?<![@\w.])(?:[a-z0-9-]+\.)+"
    r"(?:com|net|org|ar|ly|io|app|info|co|me|gob|gov|link|site|online|xyz)"
    r"(?:\.[a-z]{2})?\b(?:/[^\s]*)?"
)
_EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_CUSTOMER_ID = re.compile(r"\bCUST-\d{5}\b", re.IGNORECASE)
_OPTION_ID = re.compile(r"\bOPT-[A-Z0-9]{2,10}\b", re.IGNORECASE)
_AGREEMENT_ID = re.compile(r"\bAGR-[A-Z0-9]{4,32}\b", re.IGNORECASE)
_SECTION_ID = re.compile(r"\b(?:POL-NEG|PAY-MET|ESC|FAQ)-\d{3}\b", re.IGNORECASE)
_SCALED_NUMBER = re.compile(
    r"(?<!\w)(\d+(?:[.,]\d+)?)\s+(mil|millon(?:es)?|millón)\b", re.IGNORECASE
)
_WORD_PERCENT = re.compile(
    r"(?i)(?<![\w,.])(\d+(?:,\d+)?|[a-záéíóú]+(?:\s+y\s+[a-záéíóú]+)?)\s+por\s+ciento\b"
)
_NUMERIC_DATE = re.compile(
    r"\b(0?[1-9]|[12]\d|3[01])[/.-](0?[1-9]|1[0-2])(?:[/.-](\d{4}|\d{2}))?\b"
)
_MONTHS = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}
_TEXT_DATE = re.compile(
    rf"\b(0?[1-9]|[12]\d|3[01])\s+de\s+({'|'.join(_MONTHS)})"
    r"(?:\s+de\s+(\d{4}))?\b",
    re.IGNORECASE,
)
_WEEKDAYS = {
    "lunes": 0,
    "martes": 1,
    "miercoles": 2,
    "miércoles": 2,
    "jueves": 3,
    "viernes": 4,
    "sabado": 5,
    "sábado": 5,
    "domingo": 6,
}
_WEEKDAY_DATE = re.compile(rf"(?i)\b({'|'.join(_WEEKDAYS)})\s+(0?[1-9]|[12]\d|3[01])\b")


class ComplianceCategory(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    origin: str
    patterns: tuple[str, ...]


class ComplianceLexicon(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    categories: dict[GuardFlag, ComplianceCategory]


@lru_cache(maxsize=1)
def compliance_lexicon() -> tuple[tuple[GuardFlag, re.Pattern[str]], ...]:
    payload = yaml.safe_load(COMPLIANCE_LEXICON_PATH.read_text(encoding="utf-8"))
    lexicon = ComplianceLexicon.model_validate(payload)
    return tuple(
        (flag, re.compile(pattern))
        for flag, category in lexicon.categories.items()
        for pattern in category.patterns
    )


class ValidationContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed_numbers: tuple[str, ...] = ()
    allowed_percentages: tuple[str, ...] = ()
    allowed_customer_id: str | None = None
    allowed_option_ids: tuple[str, ...] = ()
    allowed_agreement_ids: tuple[str, ...] = ()
    cited_texts: tuple[str, ...] = ()
    high_risk: bool = False
    allowed_dates: tuple[date, ...] = ()
    current_year: int | None = None


class ValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    valid: bool
    flags: tuple[GuardFlag, ...] = ()


def _fold(text: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", text).casefold()
        if not unicodedata.combining(character)
    )


def _canonical_number(value: str) -> Decimal | None:
    raw = value.replace("$", "").replace("%", "").replace(" ", "").strip()
    if "," in raw:
        raw = raw.replace(".", "").replace(",", ".")
    elif raw.count(".") > 1 or (raw.count(".") == 1 and len(raw.rsplit(".", 1)[1]) == 3):
        raw = raw.replace(".", "")
    try:
        return Decimal(raw)
    except InvalidOperation:
        return None


def _matched_number(raw: str) -> Decimal:
    """Canonical value of a ``_NUMBER`` match, which always holds digits in es-AR form."""
    canonical = _canonical_number(raw)
    assert canonical is not None, raw
    return canonical


def _canonical_set(values: tuple[str, ...]) -> set[Decimal]:
    return {canonical for value in values if (canonical := _canonical_number(value)) is not None}


class OutputValidator:
    """Inverted validator (§10.1.4 layers 2, 3 and 5): flags anything not explicitly allowed."""

    def __init__(
        self,
        *,
        contact_allowlist: tuple[str, ...],
        prompt_canary: str | None = None,
        protected_prompt: str | None = None,
    ) -> None:
        self._contacts = frozenset(value.casefold().rstrip(".,") for value in contact_allowlist)
        self._prompt_canary = _fold(prompt_canary) if prompt_canary else None
        prompt_words = re.findall(r"[a-z0-9]+", _fold(protected_prompt or ""))
        self._prompt_ngrams = {
            " ".join(prompt_words[index : index + 8])
            for index in range(max(0, len(prompt_words) - 7))
        }

    def _contact_allowed(self, contact: str) -> bool:
        return contact.casefold().rstrip(".,") in self._contacts

    def validate(self, text: str, context: ValidationContext) -> ValidationResult:
        flags: list[GuardFlag] = []

        emails = _EMAIL.findall(text)
        without_emails = _EMAIL.sub(" ", text)
        urls = _URL.findall(without_emails)
        without_urls = _URL.sub(" ", without_emails)
        domains = _BARE_DOMAIN.findall(without_urls)
        without_domains = _BARE_DOMAIN.sub(" ", without_urls)
        phones = _PHONE.findall(without_domains)
        remaining = _PHONE.sub(" ", without_domains)
        if not all(self._contact_allowed(value) for value in (*emails, *urls, *domains, *phones)):
            flags.append("unlisted_contact")

        allowed_numbers = _canonical_set(context.allowed_numbers)
        allowed_percentages = _canonical_set(context.allowed_percentages)
        cited_numbers: set[Decimal] = set()
        cited_percentages: set[Decimal] = set()
        for cited in context.cited_texts:
            for raw in _NUMBER.findall(cited):
                canonical = _matched_number(raw)
                (cited_percentages if raw.strip().endswith("%") else cited_numbers).add(canonical)
            cited_numbers.update(numbers_in_words(cited))
        numbers_ok = allowed_numbers | cited_numbers
        percentages_ok = allowed_percentages | cited_percentages

        text_without_ids = _SECTION_ID.sub(
            " ",
            _AGREEMENT_ID.sub(" ", _OPTION_ID.sub(" ", _CUSTOMER_ID.sub(" ", remaining))),
        )

        allowed_dates = set(context.allowed_dates)
        for pattern in (_NUMERIC_DATE, _TEXT_DATE):
            for match in pattern.finditer(text_without_ids):
                day = int(match.group(1))
                month = (
                    int(match.group(2))
                    if pattern is _NUMERIC_DATE
                    else _MONTHS[_fold(match.group(2))]
                )
                raw_year = match.group(3)
                year = int(raw_year) if raw_year else context.current_year
                if year is not None and year < 100:
                    year += 2000
                try:
                    parsed_date = date(year, month, day) if year is not None else None
                except ValueError:
                    parsed_date = None
                if parsed_date is None or parsed_date not in allowed_dates:
                    flags.append("hallucinated_number")
            text_without_ids = pattern.sub(" ", text_without_ids)
        for match in _WEEKDAY_DATE.finditer(text_without_ids):
            weekday = _WEEKDAYS[_fold(match.group(1))]
            day = int(match.group(2))
            if not any(item.day == day and item.weekday() == weekday for item in allowed_dates):
                flags.append("hallucinated_number")
        text_without_ids = _WEEKDAY_DATE.sub(" ", text_without_ids)

        for match in _SCALED_NUMBER.finditer(text_without_ids):
            factor = 1000 if _fold(match.group(2)) == "mil" else 1_000_000
            if _matched_number(match.group(1)) * factor not in numbers_ok:
                flags.append("hallucinated_number")
        text_without_ids = _SCALED_NUMBER.sub(" ", text_without_ids)

        for match in _WORD_PERCENT.finditer(text_without_ids):
            amount = match.group(1)
            values: tuple[Decimal, ...] = (
                (_matched_number(amount),)
                if amount[0].isdigit()
                else numbers_in_words(f"{amount} por")
            )
            if any(value not in percentages_ok for value in values):
                flags.append("hallucinated_number")
        text_without_ids = _WORD_PERCENT.sub(" ", text_without_ids)

        for raw in _NUMBER.findall(text_without_ids):
            canonical = _matched_number(raw)
            accepted = percentages_ok if raw.strip().endswith("%") else numbers_ok
            if canonical not in accepted:
                flags.append("hallucinated_number")
                break
        if any(value not in numbers_ok for value in numbers_in_words(text_without_ids)):
            flags.append("hallucinated_number")

        customer_ids = {value.upper() for value in _CUSTOMER_ID.findall(text)}
        if customer_ids and customer_ids != {str(context.allowed_customer_id).upper()}:
            flags.append("foreign_identifier")
        option_ids = {value.upper() for value in _OPTION_ID.findall(text)}
        if option_ids - {value.upper() for value in context.allowed_option_ids}:
            flags.append("foreign_identifier")
        agreement_ids = {value.upper() for value in _AGREEMENT_ID.findall(text)}
        if agreement_ids - {value.upper() for value in context.allowed_agreement_ids}:
            flags.append("foreign_identifier")

        skeleton = detection_skeleton(text)
        for flag, pattern in compliance_lexicon():
            if pattern.search(skeleton):
                flags.append(flag)

        folded = _fold(text)
        output_words = re.findall(r"[a-z0-9]+", skeleton)
        output_ngrams = {
            " ".join(output_words[index : index + 8])
            for index in range(max(0, len(output_words) - 7))
        }
        if (
            self._prompt_canary is not None
            and (self._prompt_canary in folded or self._prompt_canary in skeleton)
        ) or self._prompt_ngrams & output_ngrams:
            flags.append("prompt_leak")

        return ValidationResult(valid=not flags, flags=tuple(dict.fromkeys(flags)))
