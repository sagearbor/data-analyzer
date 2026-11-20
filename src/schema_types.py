"""One shared mapping from the many column-type names found in data dictionaries
to the five types the quality checker understands (int, float, str, bool,
datetime). Used by BOTH the API (request validation) and the Streamlit page (the
schema it sends), so the two cannot drift. Pure functions, no I/O.

Where the names come from:
- LLM extraction (src/llm_client.py FieldDefinition / prompt): int, float, str,
  date, datetime, boolean.
- Built-in demo dictionaries (demo_dictionaries.py): string, float, date, number,
  datetime, integer, decimal, boolean (and "array", which is NOT checkable).
- PDF/regex extraction (web_app.py): date.
- REDCap exports: field types text/notes/dropdown/radio/checkbox/yesno/truefalse/
  email/phone and validation types integer/number/date_ymd/datetime_dmy/...
- Pandas/SQL/JSON-schema style names people also write: int64, float64, double,
  varchar, category, timestamp, datetime64, ...
"""
import re
from typing import Iterable, List, Optional, Tuple

CANONICAL_TYPES = ("int", "float", "str", "bool", "datetime")

SCHEMA_TYPE_ALIASES = {
    "int": ("int", "integer", "int8", "int16", "int32", "int64", "long", "bigint", "smallint", "tinyint"),
    "float": ("float", "float16", "float32", "float64", "number", "numeric", "decimal", "double", "real"),
    "str": ("str", "string", "text", "varchar", "char", "categorical", "category", "object",
            "notes", "dropdown", "radio", "checkbox", "yesno", "email", "phone", "zipcode"),
    "bool": ("bool", "boolean", "truefalse"),
    "datetime": ("datetime", "date", "time", "timestamp", "datetime64", "datetime64[ns]"),
}

_LOOKUP = {alias: canon for canon, aliases in SCHEMA_TYPE_ALIASES.items() for alias in aliases}
# REDCap date validation types: date_ymd, date_dmy, datetime_mdy, datetime_seconds_ymd, ...
_REDCAP_DATE = re.compile(r"^(date|datetime|datetime_seconds)_(ymd|dmy|mdy)$")


def normalize_schema_type(value) -> Optional[str]:
    """Canonical type for a dictionary type name (case-insensitive, trimmed),
    or None when it cannot be mapped (e.g. "array", "calc", non-strings)."""
    if not isinstance(value, str):
        return None
    key = value.strip().lower()
    if key in _LOOKUP:
        return _LOOKUP[key]
    if _REDCAP_DATE.match(key):
        return "datetime"
    return None


def normalize_schema(schema: dict) -> Tuple[dict, List[str]]:
    """Map every value of a {column: type} dict. Returns (mapped, unmapped_columns):
    columns whose type cannot be mapped are left OUT of `mapped`."""
    mapped, unmapped = {}, []
    for column, type_name in (schema or {}).items():
        canon = normalize_schema_type(type_name)
        if canon is None:
            unmapped.append(str(column))
        else:
            mapped[column] = canon
    return mapped, unmapped


def accepted_names() -> Iterable[str]:
    return sorted(_LOOKUP)
