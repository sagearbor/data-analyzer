"""
analysis_service.py — Shared core for all three transport layers (REST, MCP, A2A).

Both REST endpoints and the A2A agent delegate here. MCP tool handlers also delegate
here (after their refactor) so all three surfaces stay in sync.

IMPORTANT: This module has NO module-level heavy imports (no pandas, no mcp, no fastapi).
All heavy classes are imported lazily inside function bodies to keep this importable
under plain python3 without any installed packages.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _decode_data_content(data_content: str, encoding: str = "utf-8") -> str:
    """Decode base64 or data-URL encoded content to a plain string.

    Mirrors the exact decode/normalisation logic from the MCP tool handlers so
    all three transport layers behave identically.
    """
    try:
        if data_content.startswith("data:"):
            # Handle data URLs: "data:<mime>;base64,<data>"
            _header, data = data_content.split(",", 1)
            return base64.b64decode(data).decode(encoding)

        # Heuristic: if length is a multiple of 4 and the character set looks
        # like base64, try decoding. Accept only if the result contains a CSV
        # or TSV delimiter (avoids false-positives on short plain-text strings).
        if len(data_content) % 4 == 0 and all(
            c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
            for c in data_content
        ):
            try:
                decoded = base64.b64decode(data_content).decode(encoding)
                if "," in decoded or "\t" in decoded:
                    return decoded
            except Exception:
                pass  # Not valid base64; fall through

    except Exception:
        pass  # Any unexpected error: return content as-is

    return data_content


def _safe_import_core():
    """Lazily import the core analysis classes from mcp_server.

    Returns (DataLoader, QualityPipeline, DataDictionaryParser).
    Raises ImportError with a helpful message if mcp_server cannot be loaded.
    """
    try:
        from mcp_server import DataLoader, QualityPipeline, DataDictionaryParser  # noqa: PLC0415
        return DataLoader, QualityPipeline, DataDictionaryParser
    except ImportError as exc:
        raise ImportError(
            "Cannot import core analysis classes from mcp_server.py. "
            "Ensure pandas/numpy are installed and mcp_server.py is on the path."
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(payload: dict) -> dict:
    """Run a full data-quality analysis and return the results dict.

    Parameters
    ----------
    payload:
        JSON-serialisable dict with keys:
        - data_content (str, required): raw CSV/JSON/… text, base64, or data-URL
        - file_format (str, optional, default "csv")
        - schema (dict, optional): column-type mapping
        - rules (dict, optional): validation-rule mapping
        - min_rows (int, optional, default 1)
        - encoding (str, optional, default "utf-8")

    Returns
    -------
    dict
        The quality pipeline result dict, or ``{"error": "...", "type": "..."}``
        on bad input so every transport layer sees a consistent error shape.
    """
    try:
        DataLoader, QualityPipeline, _DDP = _safe_import_core()

        data_content: str = payload.get("data_content", "")
        if not data_content:
            return {"error": "data_content is required", "type": "validation_error"}

        file_format: str = payload.get("file_format", "csv")
        schema: dict = payload.get("schema") or {}
        rules: dict = payload.get("rules") or {}
        min_rows: int = int(payload.get("min_rows", 1))
        encoding: str = payload.get("encoding", "utf-8")

        # Decode / normalise content
        data_content = _decode_data_content(data_content, encoding)

        logger.debug("analyze: format=%s rows_min=%d", file_format, min_rows)

        df = DataLoader.load_data(data_content, file_format)
        pipeline = QualityPipeline(df, schema, rules)
        results: dict = pipeline.run_all_checks(min_rows)

        return results

    except Exception as exc:  # noqa: BLE001
        logger.error("analyze error: %s", exc, exc_info=True)
        return {"error": str(exc), "type": "analysis_error"}


def get_info(payload: dict) -> dict:
    """Return lightweight info about a dataset (shape, columns, sample rows, …).

    Parameters
    ----------
    payload:
        JSON-serialisable dict with keys:
        - data_content (str, required)
        - file_format (str, optional, default "csv")
        - sample_rows (int, optional, default 5)
        - encoding (str, optional, default "utf-8")

    Returns
    -------
    dict
        Info dict, or ``{"error": "...", "type": "..."}`` on failure.
    """
    try:
        DataLoader, _QP, _DDP = _safe_import_core()

        data_content: str = payload.get("data_content", "")
        if not data_content:
            return {"error": "data_content is required", "type": "validation_error"}

        file_format: str = payload.get("file_format", "csv")
        sample_rows: int = int(payload.get("sample_rows", 5))
        encoding: str = payload.get("encoding", "utf-8")

        data_content = _decode_data_content(data_content, encoding)

        logger.debug("get_info: format=%s sample_rows=%d", file_format, sample_rows)

        df = DataLoader.load_data(data_content, file_format)

        info: dict[str, Any] = {
            "format": file_format,
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_data": df.head(sample_rows).to_dict(orient="records"),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
        }

        return info

    except Exception as exc:  # noqa: BLE001
        logger.error("get_info error: %s", exc, exc_info=True)
        return {"error": str(exc), "type": "info_error"}
