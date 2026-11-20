#!/usr/bin/env python3
"""
Azure MCP Server for Data Analysis
Supports CSV analysis with extensibility for other data formats
Based on csvChecker functionality with future format support
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Union
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime
import os

# MCP SDK imports
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# ============================================================================
# PART 1: STANDALONE DATA QUALITY CLASSES
# ============================================================================
# These classes (DataLoader, QualityChecker, QualityPipeline) can be imported
# and used directly in any Python application without running an MCP server.
# They provide all core data validation functionality.
#
# Example usage without MCP server:
#   from mcp_server import QualityPipeline
#   pipeline = QualityPipeline(df, schema=schema_dict, rules=rules_dict)
#   results = pipeline.run_all_checks()
# ============================================================================

class DataTooLargeError(ValueError):
    """Raised when a dataset exceeds a configured size/dimension limit.

    The message is built only from our own limit names and numbers (never from
    file content), so it is safe to show to a caller.
    """


class JsonShapeError(ValueError):
    """JSON input has a shape/depth we refuse before pandas sees it. The message
    is a fixed text (never built from the content), safe to show to a caller."""


def _env_int(name: str, default: int) -> int:
    """Read a positive integer limit from the environment (default if unset/bad)."""
    try:
        value = int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


# Defaults for the dataset-shape limits (each overridable by env var; read at
# call time so a deployment can change them without re-importing this module).
DEFAULT_MAX_ROWS = 1_000_000
DEFAULT_MAX_JSON_BYTES = 25 * 1024 * 1024  # 26214400
JSON_MAX_DEPTH = 20
DEFAULT_MAX_CELLS = 50_000_000
JSON_SHAPES_MESSAGE = ("JSON data must be a list of records (every item an object) or an object of "
                       "columns (every value a list, all the same length).")
DEFAULT_MAX_COLUMNS = 2_000
DEFAULT_MAX_EXCEL_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_EXCEL_SHEETS = 50
DEFAULT_MAX_EXCEL_ARCHIVE_MEMBERS = 2_000


class DataLoader:
    """Load and validate data files (CSV, JSON, Excel, Parquet).

    Content-only by default: a ``str`` argument is ALWAYS treated as data
    (CSV/JSON text), never as a local file path, so content supplied by a
    network/agent caller can never make the server read one of its own files.
    Trusted local callers (CLI, tests) that really want to read a file use
    ``DataLoader.load_local_file(path, file_format)``, which is the only entry
    point that opens paths. MCP tool handlers, analysis_service and the A2A
    agent must never call it.
    """

    # Defense-in-depth against memory-exhaustion DoS: reject in-memory payloads
    # larger than this before handing them to pandas. Override via env var if a
    # deployment legitimately needs larger uploads.
    MAX_CONTENT_BYTES = int(os.getenv("DATA_ANALYZER_MAX_UPLOAD_BYTES", 100 * 1024 * 1024))  # 100MB

    @staticmethod
    def _check_size(content: Union[str, bytes]) -> None:
        """Raise ValueError if in-memory content exceeds the configured size cap."""
        size = len(content) if isinstance(content, (bytes, str)) else None
        if size is not None and size > DataLoader.MAX_CONTENT_BYTES:
            raise ValueError(
                f"File content ({size} bytes) exceeds the maximum allowed size "
                f"({DataLoader.MAX_CONTENT_BYTES} bytes)"
            )

    # ------------------------------------------------------------------
    # Dataset-shape limits (rows / columns / Excel decompression)
    # ------------------------------------------------------------------

    @staticmethod
    def max_rows() -> int:
        """Row cap (DATA_ANALYZER_MAX_ROWS, default 1,000,000)."""
        return _env_int("DATA_ANALYZER_MAX_ROWS", DEFAULT_MAX_ROWS)

    @staticmethod
    def max_columns() -> int:
        """Column cap (DATA_ANALYZER_MAX_COLUMNS, default 2,000)."""
        return _env_int("DATA_ANALYZER_MAX_COLUMNS", DEFAULT_MAX_COLUMNS)

    @staticmethod
    def check_dataframe_limits(df: pd.DataFrame) -> pd.DataFrame:
        """Raise DataTooLargeError if ``df`` has too many rows or columns."""
        max_rows, max_cols = DataLoader.max_rows(), DataLoader.max_columns()
        if len(df) > max_rows:
            raise DataTooLargeError(
                f"Dataset has more than the allowed {max_rows} rows "
                "(DATA_ANALYZER_MAX_ROWS)."
            )
        if len(df.columns) > max_cols:
            raise DataTooLargeError(
                f"Dataset has {len(df.columns)} columns; the allowed maximum is "
                f"{max_cols} (DATA_ANALYZER_MAX_COLUMNS)."
            )
        DataLoader.check_cells(len(df), len(df.columns))
        return df

    @staticmethod
    def max_cells() -> int:
        """Total-cells cap (DATA_ANALYZER_MAX_CELLS, default 50,000,000) = rows x columns."""
        return _env_int("DATA_ANALYZER_MAX_CELLS", DEFAULT_MAX_CELLS)

    @staticmethod
    def check_cells(rows: int, cols: int) -> None:
        """Raise DataTooLargeError when rows x columns exceeds the cells cap."""
        if rows * cols > DataLoader.max_cells():
            raise DataTooLargeError(
                f"Dataset has {rows} rows x {cols} columns, more than the allowed "
                f"{DataLoader.max_cells()} cells (DATA_ANALYZER_MAX_CELLS)."
            )

    # ------------------------------------------------------------------
    # JSON: bound BEFORE pandas builds a DataFrame
    # ------------------------------------------------------------------

    @staticmethod
    def max_json_bytes(overall_limit: Optional[int] = None) -> int:
        """JSON byte cap (DATA_ANALYZER_MAX_JSON_BYTES, default 25 MB), never above
        the overall limit (`overall_limit`, default the content-size cap)."""
        overall = overall_limit if overall_limit is not None else DataLoader.MAX_CONTENT_BYTES
        return min(_env_int("DATA_ANALYZER_MAX_JSON_BYTES", DEFAULT_MAX_JSON_BYTES), overall)

    @staticmethod
    def _json_depth_ok(obj: Any, max_depth: int = JSON_MAX_DEPTH) -> bool:
        """Iterative (no recursion) nesting-depth check."""
        stack = [(obj, 1)]
        while stack:
            node, depth = stack.pop()
            if isinstance(node, (list, dict)):
                if depth > max_depth:
                    return False
                children = node.values() if isinstance(node, dict) else node
                stack.extend((c, depth + 1) for c in children if isinstance(c, (list, dict)))
        return True

    @staticmethod
    def _count_columns(records: list, flatten_level: Optional[int], max_cols: int) -> int:
        """Columns the build will produce for a list of records, counted as the
        build names them and stopping as soon as `max_cols` is passed.

        flatten_level None: pd.DataFrame(records) -> distinct top-level keys.
        flatten_level N: pd.json_normalize(records, max_level=N) -> distinct LEAF
        PATHS: nested objects are walked while their nesting level < N (an empty
        object, a list or a scalar is a leaf). Paths are kept as tuples, so the
        count can only equal or exceed pandas' (names that collide after joining
        with "." are one column for pandas, still two paths here).
        """
        paths: set = set()
        for rec in records:
            stack = [((), rec, 0)]
            while stack:
                prefix, node, level = stack.pop()
                for key, value in node.items():
                    path = prefix + (key,)
                    if flatten_level is not None and isinstance(value, dict) and value and level < flatten_level:
                        stack.append((path, value, level + 1))
                    else:
                        paths.add(path)
                        if len(paths) > max_cols:
                            raise DataTooLargeError(
                                f"Dataset has more than the allowed {max_cols} columns (DATA_ANALYZER_MAX_COLUMNS).")
        return len(paths)

    @staticmethod
    def check_json_shape(obj: Any, flatten_level: Optional[int] = None) -> None:
        """Check a parsed JSON object BEFORE any DataFrame is built. Exactly two
        shapes are accepted: (a) a list of records (every item an object) and
        (b) an object of columns (every value a list, all the same length);
        everything else is a JsonShapeError (400). Rows, columns (counted as the
        build will name them, see _count_columns), total cells and nesting depth
        are bounded -> DataTooLargeError (413). `flatten_level` is the
        json_normalize max_level used by the build, or None when the build is
        pd.DataFrame(obj)."""
        max_rows, max_cols = DataLoader.max_rows(), DataLoader.max_columns()
        shape_error = JsonShapeError(JSON_SHAPES_MESSAGE)
        if not DataLoader._json_depth_ok(obj):
            raise JsonShapeError(f"JSON is nested more than {JSON_MAX_DEPTH} levels deep.")
        if isinstance(obj, list):
            if not all(isinstance(item, dict) for item in obj):
                raise shape_error
            rows = len(obj)
            if rows > max_rows:
                raise DataTooLargeError(
                    f"Dataset has more than the allowed {max_rows} rows (DATA_ANALYZER_MAX_ROWS).")
            cols = DataLoader._count_columns(obj, flatten_level, max_cols)
        elif isinstance(obj, dict):
            if not all(isinstance(v, list) for v in obj.values()):
                raise shape_error
            lengths = {len(v) for v in obj.values()}
            if len(lengths) > 1:
                raise shape_error
            rows, cols = (lengths.pop() if lengths else 0), len(obj)
            if cols > max_cols:
                raise DataTooLargeError(
                    f"Dataset has {cols} columns; the allowed maximum is {max_cols} (DATA_ANALYZER_MAX_COLUMNS).")
            if rows > max_rows:
                raise DataTooLargeError(
                    f"Dataset has more than the allowed {max_rows} rows (DATA_ANALYZER_MAX_ROWS).")
        else:
            raise shape_error
        DataLoader.check_cells(rows, cols)

    @staticmethod
    def build_json_dataframe(obj: Any, flatten_level: Optional[int] = None) -> pd.DataFrame:
        """Build the DataFrame from an object that passed check_json_shape (no second
        parse). Dtypes follow pd.read_json for the common cases (numeric strings ->
        numbers, all-null columns -> float NaN, date-named columns -> datetimes when
        they parse) so downstream type checks see what they saw before."""
        if isinstance(obj, list) and flatten_level is not None:
            df = pd.json_normalize(obj, max_level=flatten_level)
        else:
            df = pd.DataFrame(obj)
        for col in df.columns:
            series = df[col]
            if series.dtype != object and str(series.dtype) != "str":
                continue
            try:
                if series.isna().all():
                    df[col] = pd.to_numeric(series, errors="raise").astype("float64")
                    continue
                if series.map(lambda v: isinstance(v, (str, int, float)) and not isinstance(v, bool)).all():
                    num = series.astype("float64")  # numeric strings; raises for text
                    df[col] = num.astype("int64") if num.notna().all() and (num == num.round()).all() else num
                    continue
            except (ValueError, TypeError, OverflowError):
                pass
            if isinstance(col, str) and (col in ("modified", "date", "datetime") or col.endswith(("_at", "_time"))
                                         or col.startswith("timestamp")):
                try:
                    df[col] = pd.to_datetime(series, errors="raise")
                except (ValueError, TypeError, OverflowError):
                    pass
        return df

    @staticmethod
    def parse_json_bounded(content: Union[str, bytes], overall_limit: Optional[int] = None) -> Any:
        """Byte cap, then ONE safe parse (nesting that overflows the parser is a clean
        JsonShapeError, never a RecursionError). No shape checks."""
        limit = DataLoader.max_json_bytes(overall_limit)
        if len(content) > limit:
            raise DataTooLargeError(
                f"JSON content is larger than the allowed {limit} bytes (DATA_ANALYZER_MAX_JSON_BYTES).")
        try:
            return json.loads(content.decode("utf-8") if isinstance(content, bytes) else content)
        except RecursionError:
            raise JsonShapeError(f"JSON is nested more than {JSON_MAX_DEPTH} levels deep.")
        except (ValueError, UnicodeDecodeError):
            raise JsonShapeError("The JSON could not be parsed.")

    @staticmethod
    def precheck_json(content: Union[str, bytes], overall_limit: Optional[int] = None,
                      flatten_level: Optional[int] = None) -> Any:
        """parse_json_bounded + check_json_shape. Returns the parsed, checked object
        that the caller must build the table from (no second parse). One
        implementation for the multipart route, the JSON route and load_json."""
        obj = DataLoader.parse_json_bounded(content, overall_limit)
        DataLoader.check_json_shape(obj, flatten_level)
        return obj

    @staticmethod
    def unwrap_nested_records(obj: Any, level: int):
        """Library/MCP-only third shape (kept because DataLoader.load_json has always
        accepted it): an object holding a list of records under a key. The FIRST
        top-level key whose value is a non-empty list of objects is the table
        (flattened one level less, as before). Returns (object, flatten_level)."""
        if isinstance(obj, dict):
            for value in obj.values():
                if isinstance(value, list) and value and all(isinstance(i, dict) for i in value):
                    return value, max(level - 1, 0)
        return obj, level

    @staticmethod
    def check_excel_archive(content: bytes) -> None:
        """Refuse an .xlsx whose DECOMPRESSED size or sheet dimensions are too big.

        An .xlsx is a zip: the upload-size check only sees the compressed size,
        so a small file can expand enormously. This inspects the zip directory
        (no decompression) for the declared uncompressed sizes and member
        count, and reads only the first bytes of each worksheet for its
        ``<dimension ref="A1:Z9999"/>`` so oversize sheets are refused before
        openpyxl materialises them. Non-zip content (legacy .xls) is left to the
        post-load row/column check. Limits: DATA_ANALYZER_MAX_EXCEL_UNCOMPRESSED_BYTES,
        DATA_ANALYZER_MAX_EXCEL_SHEETS, DATA_ANALYZER_MAX_EXCEL_ARCHIVE_MEMBERS,
        DATA_ANALYZER_MAX_ROWS, DATA_ANALYZER_MAX_COLUMNS.
        """
        import re  # noqa: PLC0415
        import zipfile  # noqa: PLC0415

        if not zipfile.is_zipfile(io.BytesIO(content)):
            return
        max_bytes = _env_int(
            "DATA_ANALYZER_MAX_EXCEL_UNCOMPRESSED_BYTES", DEFAULT_MAX_EXCEL_UNCOMPRESSED_BYTES
        )
        max_sheets = _env_int("DATA_ANALYZER_MAX_EXCEL_SHEETS", DEFAULT_MAX_EXCEL_SHEETS)
        max_members = _env_int(
            "DATA_ANALYZER_MAX_EXCEL_ARCHIVE_MEMBERS", DEFAULT_MAX_EXCEL_ARCHIVE_MEMBERS
        )
        max_rows, max_cols = DataLoader.max_rows(), DataLoader.max_columns()

        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            infos = zf.infolist()
            if len(infos) > max_members:
                raise DataTooLargeError(
                    f"Excel file contains {len(infos)} internal parts; the allowed "
                    f"maximum is {max_members} (DATA_ANALYZER_MAX_EXCEL_ARCHIVE_MEMBERS)."
                )
            total = sum(i.file_size for i in infos)
            if total > max_bytes:
                raise DataTooLargeError(
                    f"Excel file expands to {total} bytes uncompressed; the allowed "
                    f"maximum is {max_bytes} (DATA_ANALYZER_MAX_EXCEL_UNCOMPRESSED_BYTES)."
                )
            sheets = [i for i in infos if re.match(r"xl/worksheets/[^/]+\.xml$", i.filename)]
            if len(sheets) > max_sheets:
                raise DataTooLargeError(
                    f"Excel file has {len(sheets)} sheets; the allowed maximum is "
                    f"{max_sheets} (DATA_ANALYZER_MAX_EXCEL_SHEETS)."
                )
            dim_re = re.compile(rb'<dimension\s+ref="([A-Za-z]+)(\d+)?(?::([A-Za-z]+)(\d+))?"')
            for info in sheets:
                with zf.open(info) as fh:
                    head = fh.read(4096)
                m = dim_re.search(head)
                if not m:
                    continue
                last_col = (m.group(3) or m.group(1)).decode().upper()
                last_row = int(m.group(4) or m.group(2) or 1)
                col_num = 0
                for ch in last_col:
                    col_num = col_num * 26 + (ord(ch) - 64)
                if last_row > max_rows or col_num > max_cols:
                    raise DataTooLargeError(
                        f"An Excel sheet spans {last_row} rows x {col_num} columns; the "
                        f"allowed maximum is {max_rows} rows x {max_cols} columns "
                        "(DATA_ANALYZER_MAX_ROWS / DATA_ANALYZER_MAX_COLUMNS)."
                    )

    @staticmethod
    def load_csv(file_path_or_content: Union[str, bytes, io.StringIO], *,
                 allow_paths: bool = False, **kwargs) -> pd.DataFrame:
        """Load CSV from bytes, text or StringIO.

        A ``str`` is CSV text unless ``allow_paths=True`` (trusted local use
        only; see ``load_local_file``), in which case an existing path is read.
        """
        try:
            DataLoader._check_size(file_path_or_content)
            max_rows = DataLoader.max_rows()
            kwargs.setdefault("nrows", max_rows + 1)  # bound memory on huge row counts
            if isinstance(file_path_or_content, bytes):
                # Decode bytes to string and create StringIO
                content = file_path_or_content.decode('utf-8')
                if not content.strip():
                    # Return empty DataFrame for empty content
                    return pd.DataFrame()
                df = pd.read_csv(io.StringIO(content), **kwargs)
            elif isinstance(file_path_or_content, str):
                if allow_paths and os.path.exists(file_path_or_content):
                    df = pd.read_csv(file_path_or_content, **kwargs)
                else:
                    # Treat as CSV content string
                    if not file_path_or_content.strip():
                        # Return empty DataFrame for empty content
                        return pd.DataFrame()
                    df = pd.read_csv(io.StringIO(file_path_or_content), **kwargs)
            else:
                df = pd.read_csv(file_path_or_content, **kwargs)
            return DataLoader.check_dataframe_limits(df)
        except DataTooLargeError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {str(e)}")
    
    @staticmethod
    def detect_encoding(file_path: str) -> str:
        """Detect file encoding for text files (trusted local paths only)"""
        import chardet
        try:
            with open(file_path, 'rb') as f:
                result = chardet.detect(f.read(100000))
                return result.get('encoding', 'utf-8')
        except:
            return 'utf-8'

    @staticmethod
    def load_json(file_path_or_content: Union[str, bytes, dict], max_depth: int = 5,
                  *, allow_paths: bool = False) -> pd.DataFrame:
        """Load JSON data and flatten nested structures up to max_depth levels.

        A ``str`` is JSON text unless ``allow_paths=True`` (trusted local use).
        """
        try:
            if isinstance(file_path_or_content, (bytes, str)):
                DataLoader._check_size(file_path_or_content)
            # Parse ONCE, bound it (bytes, depth, shape, rows, columns, cells) and
            # build the table from that same checked object.
            level = max_depth
            if isinstance(file_path_or_content, dict):
                json_data = file_path_or_content
            elif isinstance(file_path_or_content, bytes):
                json_data = DataLoader.parse_json_bounded(file_path_or_content)
            elif isinstance(file_path_or_content, str):
                if allow_paths and os.path.exists(file_path_or_content):
                    with open(file_path_or_content, 'r', encoding='utf-8') as f:
                        json_data = DataLoader.parse_json_bounded(f.read())
                else:
                    json_data = DataLoader.parse_json_bounded(file_path_or_content)
            else:
                raise ValueError("Invalid JSON input type")
            json_data, level = DataLoader.unwrap_nested_records(json_data, level)
            DataLoader.check_json_shape(json_data, level)

            df = DataLoader.build_json_dataframe(json_data, level)

            return DataLoader.check_dataframe_limits(df)
        except DataTooLargeError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to load JSON: {str(e)}")

    @staticmethod
    def load_excel(file_path_or_content: Union[str, bytes], sheet_name: Optional[str] = None,
                   *, allow_paths: bool = False) -> pd.DataFrame:
        """Load Excel file (.xlsx or .xls), combining all sheets if sheet_name not specified.

        Bytes are the normal input. A ``str`` is only accepted (as a path) with
        ``allow_paths=True`` (trusted local use). Decompressed size, sheet
        count and sheet dimensions are bounded before loading (see
        ``check_excel_archive``) and rows/columns after.
        """
        try:
            if isinstance(file_path_or_content, bytes):
                DataLoader._check_size(file_path_or_content)
                DataLoader.check_excel_archive(file_path_or_content)
                # Load from bytes
                excel_file = pd.ExcelFile(io.BytesIO(file_path_or_content))
            elif allow_paths and isinstance(file_path_or_content, str) and os.path.exists(file_path_or_content):
                # Load from file path (trusted local callers only)
                with open(file_path_or_content, "rb") as fh:
                    raw = fh.read(DataLoader.MAX_CONTENT_BYTES + 1)
                DataLoader._check_size(raw)
                DataLoader.check_excel_archive(raw)
                excel_file = pd.ExcelFile(file_path_or_content)
            else:
                raise ValueError("Excel file not found or invalid input")

            # Get all sheet names
            sheet_names = excel_file.sheet_names

            if sheet_name:
                # Load specific sheet
                if sheet_name not in sheet_names:
                    raise ValueError(f"Sheet '{sheet_name}' not found. Available sheets: {sheet_names}")
                return DataLoader.check_dataframe_limits(
                    pd.read_excel(excel_file, sheet_name=sheet_name)
                )
            else:
                # Load and combine all sheets
                dfs = []
                total_rows = 0
                for name in sheet_names:
                    df = pd.read_excel(excel_file, sheet_name=name)
                    total_rows += len(df)
                    if total_rows > DataLoader.max_rows():
                        raise DataTooLargeError(
                            f"Excel sheets together have more than the allowed "
                            f"{DataLoader.max_rows()} rows (DATA_ANALYZER_MAX_ROWS)."
                        )
                    df['_sheet_name'] = name  # Add sheet identifier
                    dfs.append(df)

                if len(dfs) == 1:
                    # Single sheet, remove sheet identifier
                    return DataLoader.check_dataframe_limits(dfs[0].drop('_sheet_name', axis=1))
                else:
                    # Multiple sheets, concatenate with sheet identifier
                    return DataLoader.check_dataframe_limits(pd.concat(dfs, ignore_index=True))
        except DataTooLargeError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to load Excel: {str(e)}")

    @staticmethod
    def load_parquet(file_path_or_content: Union[str, bytes], *, allow_paths: bool = False) -> pd.DataFrame:
        """Load Parquet from bytes (a ``str`` path only with ``allow_paths=True``)."""
        try:
            if isinstance(file_path_or_content, bytes):
                DataLoader._check_size(file_path_or_content)
                # Load from bytes
                return DataLoader.check_dataframe_limits(
                    pd.read_parquet(io.BytesIO(file_path_or_content))
                )
            elif allow_paths and isinstance(file_path_or_content, str) and os.path.exists(file_path_or_content):
                # Load from file path (trusted local callers only)
                return DataLoader.check_dataframe_limits(pd.read_parquet(file_path_or_content))
            else:
                raise ValueError("Parquet file not found or invalid input")
        except DataTooLargeError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to load Parquet: {str(e)}")

    @staticmethod
    def load_data(file_path_or_content: Union[str, bytes, io.StringIO, dict],
                  file_format: str = "csv", *, allow_paths: bool = False, **kwargs) -> pd.DataFrame:
        """Load data from various formats.

        Content-only unless ``allow_paths=True``; prefer ``load_local_file``
        for trusted local paths.
        """
        format_lower = file_format.lower()

        if format_lower == "csv":
            return DataLoader.load_csv(file_path_or_content, allow_paths=allow_paths, **kwargs)
        elif format_lower == "json":
            max_depth = kwargs.pop('max_depth', 5)
            return DataLoader.load_json(file_path_or_content, max_depth=max_depth, allow_paths=allow_paths)
        elif format_lower in ["excel", "xlsx", "xls"]:
            sheet_name = kwargs.pop('sheet_name', None)
            return DataLoader.load_excel(file_path_or_content, sheet_name=sheet_name, allow_paths=allow_paths)
        elif format_lower == "parquet":
            return DataLoader.load_parquet(file_path_or_content, allow_paths=allow_paths)
        else:
            raise ValueError(f"Unsupported file format: {file_format}. Supported: CSV, JSON, Excel, Parquet")

    @staticmethod
    def load_local_file(path: Union[str, "os.PathLike[str]"], file_format: str = "csv", **kwargs) -> pd.DataFrame:
        """TRUSTED local-file loader for the CLI / Streamlit app / tests.

        Reads ``path`` from this machine's filesystem. Never pass a value that
        came from a remote caller (MCP tool arguments, A2A payloads, HTTP
        requests) - those must go through the content-only loaders above.
        """
        path_str = os.fspath(path)
        if not os.path.isfile(path_str):
            raise ValueError(f"File not found: {os.path.basename(path_str)}")
        return DataLoader.load_data(path_str, file_format, allow_paths=True, **kwargs)

class DataDictionaryParser:
    """Parse REDCap-style data dictionaries to extract schema and validation rules"""

    @staticmethod
    def parse_redcap_dictionary(dict_df: pd.DataFrame) -> tuple[Dict[str, str], Dict[str, Dict]]:
        """
        Parse REDCap data dictionary CSV to extract schema types and validation rules.

        Returns:
            tuple: (schema_dict, rules_dict)
                - schema_dict: {field_name: data_type}
                - rules_dict: {field_name: {allowed: [...], min: X, max: Y}}
        """
        schema = {}
        rules = {}

        # Expected column names (with variations)
        field_col = None
        type_col = None
        choices_col = None
        validation_col = None
        min_col = None
        max_col = None

        # Find the right column names (case-insensitive)
        cols_lower = {col.lower(): col for col in dict_df.columns}

        for key in ['variable / field name', 'variable', 'field name', 'variable_field_name']:
            if key in cols_lower:
                field_col = cols_lower[key]
                break

        for key in ['field type', 'type', 'field_type']:
            if key in cols_lower:
                type_col = cols_lower[key]
                break

        for key in ['choices, calculations, or slider labels', 'choices', 'field_choices']:
            if key in cols_lower:
                choices_col = cols_lower[key]
                break

        for key in ['text validation type or show slider number', 'validation', 'text_validation_type_or_show_slider_number']:
            if key in cols_lower:
                validation_col = cols_lower[key]
                break

        for key in ['text validation min', 'min', 'text_validation_min']:
            if key in cols_lower:
                min_col = cols_lower[key]
                break

        for key in ['text validation max', 'max', 'text_validation_max']:
            if key in cols_lower:
                max_col = cols_lower[key]
                break

        if not field_col:
            raise ValueError("Could not find field name column in data dictionary")

        # Parse each field
        for idx, row in dict_df.iterrows():
            field_name = row[field_col]
            if pd.isna(field_name) or field_name == '':
                continue

            # Extract field type
            field_type = row[type_col] if type_col and not pd.isna(row[type_col]) else 'text'

            # Map REDCap types to our types
            if field_type in ['text', 'notes']:
                # Check if there's a validation type
                if validation_col and not pd.isna(row[validation_col]):
                    validation = str(row[validation_col]).lower()
                    if 'date' in validation:
                        schema[field_name] = 'datetime'
                    elif 'number' in validation or 'integer' in validation:
                        if 'integer' in validation:
                            schema[field_name] = 'int'
                        else:
                            schema[field_name] = 'float'
                    else:
                        schema[field_name] = 'str'
                else:
                    schema[field_name] = 'str'
            elif field_type in ['radio', 'dropdown', 'checkbox']:
                # These have allowed values in the choices column
                schema[field_name] = 'int'  # Usually coded as integers

                # Extract allowed values from choices
                if choices_col and not pd.isna(row[choices_col]):
                    choices_str = str(row[choices_col])
                    # Parse format: "1, Male | 2, Female" or "1, Yes | 0, No"
                    allowed_values = []
                    for choice in choices_str.split('|'):
                        choice = choice.strip()
                        if ',' in choice:
                            code = choice.split(',')[0].strip()
                            try:
                                # Try to convert to int
                                allowed_values.append(int(code))
                            except ValueError:
                                # If not int, store as string
                                allowed_values.append(code)

                    if allowed_values:
                        # For checkbox fields, REDCap expands them to field___1, field___2, etc.
                        # Each checkbox column can only be 0 (unchecked) or 1 (checked)
                        if field_type == 'checkbox':
                            # Create rules for each expanded checkbox field
                            for code in allowed_values:
                                checkbox_field = f"{field_name}___{code}"
                                schema[checkbox_field] = 'int'
                                rules[checkbox_field] = {"allowed": [0, 1]}
                        else:
                            # Radio/dropdown: use original field name with full allowed values
                            rules[field_name] = {"allowed": allowed_values}
            elif field_type == 'calc':
                schema[field_name] = 'float'
            elif field_type == 'yesno':
                schema[field_name] = 'int'
                rules[field_name] = {"allowed": [0, 1]}
            elif field_type == 'truefalse':
                schema[field_name] = 'bool'

            # Extract min/max ranges
            if min_col and not pd.isna(row[min_col]):
                try:
                    min_val = float(row[min_col])
                    if field_name not in rules:
                        rules[field_name] = {}
                    rules[field_name]["min"] = min_val
                except (ValueError, TypeError):
                    pass

            if max_col and not pd.isna(row[max_col]):
                try:
                    max_val = float(row[max_col])
                    if field_name not in rules:
                        rules[field_name] = {}
                    rules[field_name]["max"] = max_val
                except (ValueError, TypeError):
                    pass

        return schema, rules

class QualityChecker:
    """Perform data quality checks on structured data (CSV, future: other formats)"""

    def __init__(self, df: pd.DataFrame, schema: Optional[Dict] = None, rules: Optional[Dict] = None):
        self.df = df
        self.schema = schema or {}
        self.rules = rules or {}
        self.issues = []
        
    def check_row_count(self, min_rows: int = 1) -> Dict[str, Any]:
        """Check if DataFrame has minimum required rows"""
        row_count = len(self.df)
        passed = row_count >= min_rows
        
        result = {
            "check": "row_count",
            "passed": passed,
            "row_count": row_count,
            "min_required": min_rows,
            "message": f"Found {row_count} rows (minimum: {min_rows})"
        }
        
        if not passed:
            self.issues.append(result)
            
        return result
    
    def check_data_types(self) -> Dict[str, Any]:
        """Validate column data types against schema or auto-detected types"""
        # Use provided schema OR auto-detected types for validation
        auto_detected_types = self._auto_detect_types()
        
        # Combine manual schema with auto-detected types for validation
        validation_schema = auto_detected_types.copy()
        if self.schema:
            validation_schema.update(self.schema)  # Manual schema overrides auto-detection
        
        if not validation_schema:
            return {
                "check": "data_types",
                "passed": True,
                "message": "No schema or auto-detection available, skipping type validation"
            }
        
        type_issues = []
        total_checks = 0
        
        for column, expected_type in validation_schema.items():
            if column not in self.df.columns:
                type_issues.append({
                    "column": column,
                    "issue": "missing_column",
                    "expected_type": expected_type
                })
                continue
                
            total_checks += 1
            actual_dtype = str(self.df[column].dtype)
            
            # Type mapping
            type_map = {
                "int": ["int64", "int32", "int16", "int8"],
                "float": ["float64", "float32"],
                "str": ["object", "string"],
                "bool": ["bool"],
                "datetime": ["datetime64[ns]"]
            }
            
            if expected_type in type_map:
                if actual_dtype not in type_map[expected_type]:
                    # Try to convert and see if it fails
                    try:
                        if expected_type == "int":
                            pd.to_numeric(self.df[column], errors='raise')
                        elif expected_type == "float":
                            pd.to_numeric(self.df[column], errors='raise')
                        elif expected_type == "datetime":
                            import warnings
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                col_data = self.df[column].dropna()
                                parsed_dates = pd.to_datetime(col_data, errors='coerce')
                            # Count how many values failed to parse (became NaT)
                            failed_count = parsed_dates.isna().sum()
                            # If any values failed to parse, this is a type error
                            if failed_count > 0:
                                raise ValueError(f"Invalid datetime values found: {failed_count} values could not be parsed")
                        elif expected_type == "bool":
                            # Object column that should be boolean: every non-null value
                            # must look like a boolean.
                            col_vals = self.df[column].dropna()
                            bad = ~col_vals.astype(str).str.strip().str.lower().isin(self._BOOL_WORDS)
                            if bad.any():
                                raise ValueError("Non-boolean values found")
                    except (ValueError, TypeError):
                        # For datetime issues, provide more specific information
                        if expected_type == "datetime":
                            col_data = self.df[column].dropna()
                            import warnings
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                invalid_dates = col_data[pd.to_datetime(col_data, errors='coerce').isna()]
                            type_issues.append({
                                "column": column,
                                "issue": "datetime_validation_failed",
                                "expected_type": expected_type,
                                "actual_type": actual_dtype,
                                "invalid_values": invalid_dates.tolist(),
                                "violating_rows": invalid_dates.index.tolist()[:10],  # first 10
                                "violation_count": int(len(invalid_dates)),
                                "valid_sample": col_data[pd.to_datetime(col_data, format='mixed', errors='coerce').notna()].head(2).tolist(),
                                "description": f"Found {len(invalid_dates)} invalid date values in column '{column}'"
                            })
                        else:
                            bad_rows = self._type_violating_rows(column, expected_type)
                            type_issues.append({
                                "column": column,
                                "issue": "type_mismatch",
                                "expected_type": expected_type,
                                "actual_type": actual_dtype,
                                "sample_values": self.df[column].head(3).tolist(),
                                "violating_rows": bad_rows[:10],  # first 10
                                "violation_count": len(bad_rows),
                            })
        
        passed = len(type_issues) == 0
        result = {
            "check": "data_types",
            "passed": passed,
            "total_columns_checked": total_checks,
            "issues_found": len(type_issues),
            "issues": type_issues,
            "message": f"Type validation: {len(type_issues)} issues found"
        }
        
        if not passed:
            self.issues.extend(type_issues)
            
        return result
    
    _RANGE_RULE_KEYS = ("min", "max", "allowed")
    _BOOL_WORDS = frozenset({"true", "false", "t", "f", "yes", "no", "y", "n", "1", "0", "1.0", "0.0"})

    def _type_violating_rows(self, column: str, expected_type: str) -> list:
        """Index labels of the non-null values that do not fit an int/float/bool column."""
        col = self.df[column]
        if expected_type in ("int", "float"):
            bad = col.notna() & pd.to_numeric(col, errors="coerce").isna()
        elif expected_type == "bool":
            bad = col.notna() & ~col.astype(str).str.strip().str.lower().isin(self._BOOL_WORDS)
        else:
            return []
        return col.index[bad].tolist()

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        return (isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, (bool, np.bool_))
                and bool(np.isfinite(value)))

    @staticmethod
    def _sorted_for_display(values: Any) -> list:
        """Sorted copy for the rule text; falls back to string order for mixed types."""
        try:
            return sorted(values)
        except TypeError:
            return sorted(values, key=str)

    def _normalise_range_rule(self, raw_rule: Any):
        """Split one column's rule into (applicable_rule, problems).

        problems is a list of (key, reason) with stable reason codes
        (rule_not_object, unknown_key, min_not_numeric, max_not_numeric,
        allowed_not_list, allowed_item_not_scalar). Only the keys that are
        well-formed are returned in applicable_rule.
        """
        if not isinstance(raw_rule, dict):
            return {}, [(None, "rule_not_object")]
        rule: Dict[str, Any] = {}
        problems = []
        for key, value in raw_rule.items():
            if key not in self._RANGE_RULE_KEYS:
                problems.append((str(key)[:60], "unknown_key"))
            elif key in ("min", "max"):
                if self._is_finite_number(value):
                    rule[key] = value
                else:
                    problems.append((key, f"{key}_not_numeric"))
            else:  # allowed
                if not isinstance(value, (list, tuple, set, frozenset)):
                    problems.append((key, "allowed_not_list"))
                elif not all(v is None or isinstance(v, (str, int, float, bool, np.integer, np.floating))
                             for v in value):
                    problems.append((key, "allowed_item_not_scalar"))
                else:
                    rule[key] = list(value)
        return rule, problems

    def check_value_ranges(self) -> Dict[str, Any]:
        """Check if values fall within specified ranges/sets"""
        if not self.rules:
            return {
                "check": "value_ranges",
                "passed": True,
                "message": "No rules provided, skipping range validation"
            }
        
        range_issues = []
        
        for column, raw_rule in self.rules.items():
            # A rule that cannot be applied is reported, never silently dropped.
            rule, rule_problems = self._normalise_range_rule(raw_rule)
            for key, reason in rule_problems:
                range_issues.append({
                    "column": column,
                    "rule": "rule_invalid",
                    "issue": "rule_invalid",
                    "key": key,
                    "reason": reason,
                    "violation_count": 0,
                })

            if column not in self.df.columns:
                continue
                
            col_data = self.df[column].dropna()
            
            # Check numeric ranges
            if "min" in rule or "max" in rule:
                try:
                    numeric_data = pd.to_numeric(col_data, errors='coerce')
                    
                    if "min" in rule:
                        violations = numeric_data < rule["min"]
                        if violations.any():
                            violating_rows = numeric_data[violations].index.tolist()
                            range_issues.append({
                                "column": column,
                                "rule": f"min >= {rule['min']}",
                                "violation_count": violations.sum(),
                                "violating_rows": violating_rows[:10]  # First 10
                            })
                    
                    if "max" in rule:
                        violations = numeric_data > rule["max"]
                        if violations.any():
                            violating_rows = numeric_data[violations].index.tolist()
                            range_issues.append({
                                "column": column,
                                "rule": f"max <= {rule['max']}",
                                "violation_count": violations.sum(),
                                "violating_rows": violating_rows[:10]  # First 10
                            })
                except (TypeError, ValueError):
                    # The bounds were validated above, so this is not expected;
                    # if it ever happens the rule must still be visible.
                    range_issues.append({
                        "column": column,
                        "rule": "rule_invalid",
                        "issue": "rule_invalid",
                        "key": "min/max",
                        "reason": "comparison_failed",
                        "violation_count": 0,
                    })
            
            # Check allowed values
            if "allowed" in rule:
                # Normalize allowed values - convert to numeric if possible
                allowed_values = set()
                for val in rule["allowed"]:
                    try:
                        # Try numeric conversion
                        allowed_values.add(int(val))
                        allowed_values.add(float(val))
                    except (ValueError, TypeError):
                        pass
                    # Also add original value
                    allowed_values.add(val)

                # Normalize actual values - convert to numeric if possible
                normalized_actual = []
                for val in col_data.unique():
                    try:
                        # Try to convert to numeric
                        numeric_val = pd.to_numeric(val, errors='raise')
                        normalized_actual.append(numeric_val)
                    except (ValueError, TypeError):
                        # Keep as-is if not numeric
                        normalized_actual.append(val)

                actual_values = set(normalized_actual)
                invalid_values = actual_values - allowed_values

                if invalid_values:
                    # Find rows with invalid values
                    # Build mask based on full dataframe
                    invalid_mask = pd.Series([False] * len(self.df), index=self.df.index)

                    for idx in col_data.index:
                        val = col_data.loc[idx]
                        try:
                            normalized = pd.to_numeric(val, errors='raise')
                        except (ValueError, TypeError):
                            normalized = val
                        if normalized in invalid_values:
                            invalid_mask.loc[idx] = True

                    violating_rows = self.df[invalid_mask].index.tolist()

                    range_issues.append({
                        "column": column,
                        "rule": f"allowed_values: {self._sorted_for_display(rule['allowed'])}",
                        "invalid_values": sorted([str(v) for v in invalid_values]),
                        "violation_count": invalid_mask.sum(),
                        "violating_rows": violating_rows[:10]  # First 10
                    })
        
        passed = len(range_issues) == 0
        result = {
            "check": "value_ranges",
            "passed": passed,
            "issues_found": len(range_issues),
            "issues": range_issues,
            "message": f"Range validation: {len(range_issues)} issues found"
        }
        
        if not passed:
            self.issues.extend(range_issues)
            
        return result
    
    def _auto_detect_types(self) -> Dict[str, str]:
        """Auto-detect column types including dates"""
        detected_types = {}
        
        for col in self.df.columns:
            col_data = self.df[col].dropna()
            if len(col_data) == 0:
                detected_types[col] = "unknown"
                continue
                
            # Check if it's numeric
            try:
                numeric_data = pd.to_numeric(col_data, errors='coerce')
                if not numeric_data.isna().all():
                    # Check if all values are integers
                    if (numeric_data % 1 == 0).all():
                        detected_types[col] = "int"
                    else:
                        detected_types[col] = "float"
                    continue
            except:
                pass
            
            # Check if it's datetime
            try:
                # Try to parse as datetime, but only if it has valid-looking patterns
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    datetime_parsed = pd.to_datetime(col_data, errors='coerce')
                
                # Check if more than 50% of non-null values were successfully parsed
                # and at least one value looks like a date
                valid_dates = datetime_parsed.notna().sum()
                if valid_dates > len(col_data) * 0.5 and valid_dates > 0:
                    # Double-check with strict parsing to catch invalid dates
                    sample_values = col_data.head(3).tolist()
                    if any(any(char in str(val) for char in ['-', '/', ':', ' ']) for val in sample_values):
                        detected_types[col] = "datetime"
                        continue
            except:
                pass
            
            # Check if it's boolean
            unique_values = set(str(v).lower() for v in col_data.unique())
            if unique_values.issubset({'true', 'false', '1', '0', 'yes', 'no', 't', 'f', 'y', 'n'}):
                detected_types[col] = "bool"
                continue
            
            # Default to string
            detected_types[col] = "str"
        
        return detected_types
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics for the dataset"""
        stats = {
            "shape": {
                "rows": len(self.df),
                "columns": len(self.df.columns)
            },
            "columns": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "auto_detected_types": self._auto_detect_types(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "duplicate_rows": self.df.duplicated().sum(),
            "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
        
        # Add basic stats for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats["numeric_summary"] = {}
            for col in numeric_cols:
                stats["numeric_summary"][col] = {
                    "min": float(self.df[col].min()) if not pd.isna(self.df[col].min()) else None,
                    "max": float(self.df[col].max()) if not pd.isna(self.df[col].max()) else None,
                    "mean": float(self.df[col].mean()) if not pd.isna(self.df[col].mean()) else None,
                    "std": float(self.df[col].std()) if not pd.isna(self.df[col].std()) else None
                }
        
        return stats

class QualityPipeline:
    """Orchestrate all quality checks"""
    
    def __init__(self, df: pd.DataFrame, schema: Optional[Dict] = None, rules: Optional[Dict] = None):
        self.checker = QualityChecker(df, schema, rules)
        
    def run_all_checks(self, min_rows: int = 1) -> Dict[str, Any]:
        """Run all quality checks and return comprehensive results"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "summary_stats": self.checker.get_summary_stats(),
            "checks": {},
            "overall_passed": True,
            "total_issues": 0
        }
        
        # Run individual checks
        row_count_result = self.checker.check_row_count(min_rows)
        type_check_result = self.checker.check_data_types()
        range_check_result = self.checker.check_value_ranges()
        
        results["checks"]["row_count"] = row_count_result
        results["checks"]["data_types"] = type_check_result
        results["checks"]["value_ranges"] = range_check_result
        
        # Determine overall status
        all_checks_passed = all([
            row_count_result["passed"],
            type_check_result["passed"],
            range_check_result["passed"]
        ])
        
        results["overall_passed"] = all_checks_passed
        results["total_issues"] = len(self.checker.issues)
        results["issues"] = self.checker.issues

        return results

# ============================================================================
# PART 2: MCP SERVER WRAPPER (OPTIONAL)
# ============================================================================
# This section wraps the standalone classes above as an MCP (Model Context Protocol)
# server for external access by AI assistants and other MCP clients.
#
# The classes above work fine WITHOUT this server running - this is only needed
# if you want to expose the validation functionality via MCP protocol.
#
# To run as MCP server:
#   python mcp_server.py
#
# To use without MCP server (recommended for web_app.py):
#   Just import the classes directly (see PART 1 comments above)
# ============================================================================

app = Server("data-analyzer")

@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="analyze_data",
            description="Analyze structured data for quality issues, data types, and validation (CSV supported, extensible)",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_content": {
                        "type": "string",
                        "description": "Data content as string or base64 encoded data"
                    },
                    "file_format": {
                        "type": "string",
                        "description": "Data format (csv, future: json, excel)",
                        "default": "csv"
                    },
                    "schema": {
                        "type": "object",
                        "description": "Expected data types for columns (e.g., {'id': 'int', 'name': 'str'})",
                        "additionalProperties": {"type": "string"}
                    },
                    "rules": {
                        "type": "object",
                        "description": "Validation rules (e.g., {'age': {'min': 0, 'max': 120}, 'country': {'allowed': ['USA', 'CAN']}})",
                        "additionalProperties": {"type": "object"}
                    },
                    "min_rows": {
                        "type": "integer",
                        "description": "Minimum required number of rows",
                        "default": 1
                    },
                    "encoding": {
                        "type": "string",
                        "description": "Data encoding (utf-8, latin1, etc.)",
                        "default": "utf-8"
                    }
                },
                "required": ["data_content"]
            }
        ),
        types.Tool(
            name="get_data_info",
            description="Get basic information about structured data without validation (CSV supported, extensible)",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_content": {
                        "type": "string",
                        "description": "Data content as string or base64 encoded data"
                    },
                    "file_format": {
                        "type": "string",
                        "description": "Data format (csv, future: json, excel)",
                        "default": "csv"
                    },
                    "sample_rows": {
                        "type": "integer",
                        "description": "Number of sample rows to return",
                        "default": 5
                    }
                },
                "required": ["data_content"]
            }
        )
    ]

@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls by delegating to the shared analysis_service module.

    Both ``analyze_data`` and ``get_data_info`` delegate to ``analysis_service``
    so the business logic is defined in exactly one place and all three transport
    layers (REST, MCP, A2A) stay in sync automatically.
    """
    # Lazy import avoids a circular dependency at module load time.
    import analysis_service  # noqa: PLC0415

    if name == "analyze_data":
        payload = {
            "data_content": arguments.get("data_content", ""),
            "file_format": arguments.get("file_format", "csv"),
            "schema": arguments.get("schema", {}),
            "rules": arguments.get("rules", {}),
            "min_rows": arguments.get("min_rows", 1),
            "encoding": arguments.get("encoding", "utf-8"),
        }
        result = analysis_service.analyze(payload)
        return [
            types.TextContent(
                type="text",
                text=json.dumps(result, indent=2, default=str),
            )
        ]

    elif name == "get_data_info":
        payload = {
            "data_content": arguments.get("data_content", ""),
            "file_format": arguments.get("file_format", "csv"),
            "sample_rows": arguments.get("sample_rows", 5),
            "encoding": arguments.get("encoding", "utf-8"),
        }
        result = analysis_service.get_info(payload)
        return [
            types.TextContent(
                type="text",
                text=json.dumps(result, indent=2, default=str),
            )
        ]

    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    """Main entry point for the MCP server"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="data-analyzer",
                server_version="1.0.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
