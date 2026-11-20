"""
Data Quality Analyzer - Clean UI with Proper Layout
Fixed navbar, three-column layout, no sidebar issues
"""

import streamlit as st
import pandas as pd
import json
import base64
import io
import asyncio
import csv
import requests
from typing import Dict, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pypdf
import re
import plotly.graph_objects as go
import plotly.express as px
import hashlib
import os
import tempfile
import subprocess
import sys
import stat as stat_module
import time
import logging
import traceback
from pathlib import Path

# Import custom modules
from demo_dictionaries import DEMO_DICTIONARIES, get_demo_dictionary
from src.__version__ import __version__
from src.runtime_env import ApiConfigError, resolve_api_base_url
from src.schema_types import normalize_schema, normalize_schema_type

logger = logging.getLogger("data_analyzer.ui")
# NOTE: QualityPipeline/QualityChecker (mcp_server.py) are no longer imported
# here. Rule-based quality analysis now runs centrally behind the REST API
# (POST /api/v1/analyze) so the UI, the API, and the MCP server all share one
# running backend instead of the UI running the engine in-process. See
# DataQualityAnalyzer below.
#
# DataLoader IS still imported and used locally for parsing/previewing
# uploaded Excel (.xlsx/.xls) files client-side (e.g. sheet peeking, dictionary
# ingestion) before data is handed off to DataQualityAnalyzer, which sends it
# to the API for analysis.
from mcp_server import DataLoader
# Force use of custom renderer for better compatibility
MERMAID_AVAILABLE = False
from mermaid_renderer import render_mermaid
from src.api_examples import EXAMPLE_GROUPS, build_api_examples, display_base_url

# Import LLM parser
try:
    from src.llm_client import LLMDictionaryParser, get_available_deployments, resolve_deployment_choice
    from src.dcri_llm import llm_action, set_user
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("LLM client not available. Install openai package to enable.")


def llm_model_label() -> str:
    """Name of the Azure OpenAI deployment the UI will use by default, read
    from config (AZURE_OPENAI_DEPLOYMENTS / AZURE_OPENAI_DEPLOYMENT) - never a
    hard-coded model name, so the label can't drift from what is deployed."""
    if LLM_AVAILABLE:
        deployments = get_available_deployments()
        if deployments:
            return deployments[0]
    return os.getenv("AZURE_OPENAI_DEPLOYMENT") or "deployment not configured"

# Import Named Program Cache and Logic Validation Engine
try:
    from src.program_manager import ProgramManager
    from src.logic_engine import LogicValidator, RuleExtractor, ConditionalRule
    PROGRAM_CACHE_AVAILABLE = True
except ImportError:
    PROGRAM_CACHE_AVAILABLE = False
    logger.warning("Program cache and logic validation not available.")

# Import environment banner
try:
    import envbanner
    ENVBANNER_AVAILABLE = True
except ImportError:
    ENVBANNER_AVAILABLE = False
    logger.warning("env-banner not available. Install with: pip install env-banner")

# ---------------------------------------------------------------------------
# Security helpers (release 1.0 hardening). Pure functions / small classes so
# they can be unit tested without a running Streamlit session
# (tests/test_sec_ui_*.py).
# ---------------------------------------------------------------------------
import secrets
import string as _string


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Positive integer from the environment; bad/blank/too-small values fall
    back to `default` so a typo can never switch a limit off."""
    try:
        value = int((os.environ.get(name) or "").strip())
    except ValueError:
        return default
    return value if value >= minimum else default


# Upload / dictionary limits (all overridable by environment variable).
DEFAULT_MAX_UPLOAD_MB = 25          # also enforced by Streamlit (server.maxUploadSize)
DEFAULT_PDF_MAX_PAGES = 200
DEFAULT_DICT_MAX_CHARS = 500_000
DEFAULT_CACHE_TTL_HOURS = 12
_LINE_CAP = 500                     # per-line cap for the manual PDF patterns


def max_upload_bytes() -> int:
    return _env_int("UI_MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB) * 1024 * 1024


class DictionaryTooLargeError(ValueError):
    """The dictionary is over a configured size limit (message is user-safe)."""


_MD_PUNCT = re.compile("([" + re.escape(_string.punctuation) + "])")


def md_escape(value) -> str:
    """Make any untrusted value render as plain text through st.markdown /
    st.error / st.warning (which interpret markdown, links, images and $math$).

    Every ASCII punctuation character is backslash-escaped, which CommonMark
    renders as the literal character, so `![x](https://evil/p.png)` is shown
    as text and never fetched. Newlines are kept.
    """
    return _MD_PUNCT.sub(r"\\\1", str(value))


def file_fingerprint(name: str, size: Optional[int] = None) -> str:
    """Log-safe description of an uploaded file: extension, size and a short
    SHA-256 of the name. The name itself (it can encode study/site ids) is
    never logged."""
    name = str(name or "")
    ext = name.rsplit(".", 1)[-1].lower()[:8] if "." in name else "none"
    digest = hashlib.sha256(name.encode("utf-8", "replace")).hexdigest()[:8]
    return f"ext={ext} size={size if size is not None else 'unknown'} name_sha256={digest}"


# --- Excel export: formula neutralisation (B-02) ---------------------------
_FORMULA_PREFIXES = ("=", "+", "-", "@")


def looks_like_formula(value) -> bool:
    """True for text a spreadsheet could execute: starts with = + - @ even
    after leading whitespace / tab / CR / LF."""
    return isinstance(value, str) and value.lstrip(" \t\r\n\v\f ").startswith(_FORMULA_PREFIXES)


def force_text_cells(workbook) -> int:
    """Turn every formula-looking string cell in `workbook` into a literal text
    cell (data_type 's', plus Excel's quotePrefix so it stays text when edited).
    Real numbers/dates/booleans are never touched. Returns the cell count."""
    changed = 0
    for ws in workbook.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if looks_like_formula(cell.value):
                    cell.data_type = "s"
                    cell.quotePrefix = True
                    changed += 1
    return changed


# --- Dictionary text limits (B-04) -----------------------------------------
DEFAULT_PDF_EXTRACT_TIMEOUT_SECONDS = 30
DEFAULT_PDF_EXTRACT_MAX_MEMORY_MB = 400
_PDF_WORKER_PATH = Path(__file__).resolve().parent / "src" / "pdf_extract_worker.py"
# Only these variables reach the PDF child: no API key, no Azure settings.
_PDF_CHILD_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT",
                       "PYTHONUSERBASE", "PYTHONPATH", "VIRTUAL_ENV")


def _pdf_worker_command(max_pages: int, max_chars: int, max_memory_mb: int) -> list:
    """The child command line. A separate function so tests can substitute the worker."""
    return [sys.executable, str(_PDF_WORKER_PATH), str(max_pages), str(max_chars), str(max_memory_mb)]


def _run_pdf_worker(pdf_bytes: bytes, max_pages: int, max_chars: int) -> dict:
    """Run the extraction worker in a child process with a wall-clock timeout, a
    memory ceiling (applied in the child, Linux only - see pdf_extract_worker) and
    a cap on the bytes read from its stdout. Returns the worker's JSON result;
    raises DictionaryTooLargeError (user-safe message) for every failure - a
    partial result is never returned."""
    import threading

    timeout = _env_int("UI_PDF_EXTRACT_TIMEOUT_SECONDS", DEFAULT_PDF_EXTRACT_TIMEOUT_SECONDS)
    mem_mb = _env_int("UI_PDF_EXTRACT_MAX_MEMORY_MB", DEFAULT_PDF_EXTRACT_MAX_MEMORY_MB)
    out_cap = max_chars * 12 + 4096  # JSON-escaped text is at most 12 bytes per character
    env = {k: os.environ[k] for k in _PDF_CHILD_ENV_KEYS if k in os.environ}
    unreadable = DictionaryTooLargeError(
        "This PDF could not be read safely (the reader stopped or produced invalid output). "
        "Re-save it as a standard PDF, or split it, and upload it again.")
    try:
        proc = subprocess.Popen(_pdf_worker_command(max_pages, max_chars, mem_mb),
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                env=env, close_fds=True, shell=False)
    except OSError:
        logger.error("pdf_worker_start_failed")
        raise unreadable

    box: list = []

    def _write():
        try:
            proc.stdin.write(pdf_bytes)
        except OSError:
            pass
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

    def _read():
        box.append(proc.stdout.read(out_cap + 1))

    threads = [threading.Thread(target=_write, daemon=True), threading.Thread(target=_read, daemon=True)]
    for t in threads:
        t.start()
    deadline = time.monotonic() + timeout
    timed_out = False
    while proc.poll() is None:
        if time.monotonic() > deadline:
            timed_out = True
            break
        if box and len(box[0]) > out_cap:  # over-long output: stop the child
            break
        time.sleep(0.02)
    if proc.poll() is None:
        proc.kill()
    proc.wait()
    for t in threads:
        t.join(timeout=2)
    if timed_out:
        logger.warning("pdf_worker_timeout seconds=%d", timeout)
        raise DictionaryTooLargeError(
            f"Reading this PDF took too long (more than {timeout} seconds) and was stopped. "
            "Split the dictionary into smaller files and upload them separately.")
    out = box[0] if box else b""
    if proc.returncode != 0 or not out or len(out) > out_cap:
        logger.warning("pdf_worker_failed returncode=%s output_bytes=%d", proc.returncode, len(out))
        raise unreadable
    try:
        result = json.loads(out.decode("utf-8"))
        if not isinstance(result, dict) or result.get("status") not in (
                "ok", "too_many_pages", "too_much_text", "error"):
            raise ValueError("unexpected worker output")
    except (ValueError, UnicodeDecodeError):
        logger.warning("pdf_worker_bad_output")
        raise unreadable
    return result


def extract_pdf_text_and_pages(pdf_bytes: bytes, max_pages: Optional[int] = None,
                               max_chars: Optional[int] = None) -> tuple:
    """Extract text from PDF bytes in a bounded child process.

    Returns (text, page_count). Raises DictionaryTooLargeError (message fit to
    show the user) when the PDF has more than `max_pages` pages or more than
    `max_chars` characters of text, takes too long, exceeds the child's memory
    ceiling, or cannot be read. Defaults: UI_PDF_MAX_PAGES / UI_DICT_MAX_CHARS.
    """
    max_pages = max_pages or _env_int("UI_PDF_MAX_PAGES", DEFAULT_PDF_MAX_PAGES)
    max_chars = max_chars or _env_int("UI_DICT_MAX_CHARS", DEFAULT_DICT_MAX_CHARS)
    result = _run_pdf_worker(pdf_bytes, max_pages, max_chars)
    status = result.get("status")
    if status == "too_many_pages":
        raise DictionaryTooLargeError(
            f"This PDF has {int(result.get('pages', 0))} pages; the limit is {max_pages}. "
            "Split the dictionary into smaller files and upload them separately."
        )
    if status == "too_much_text":
        raise DictionaryTooLargeError(
            f"The text in this PDF is longer than the limit of {max_chars:,} characters. "
            "Split the dictionary into smaller files and upload them separately."
        )
    if status != "ok" or not isinstance(result.get("text"), str):
        if result.get("reason") == "memory":
            raise DictionaryTooLargeError(
                "This PDF needs too much memory to read safely. "
                "Split the dictionary into smaller files and upload them separately.")
        raise DictionaryTooLargeError(
            "This PDF could not be read. Re-save it as a standard PDF and upload it again.")
    return result["text"], int(result.get("pages", 0))


def extract_pdf_text(pdf_bytes: bytes, max_pages: Optional[int] = None, max_chars: Optional[int] = None) -> str:
    """Text of a PDF (see extract_pdf_text_and_pages for the limits and errors)."""
    return extract_pdf_text_and_pages(pdf_bytes, max_pages, max_chars)[0]


def check_dictionary_text_size(text: str, max_chars: Optional[int] = None) -> str:
    """Same character limit for non-PDF dictionary text sent to the LLM."""
    max_chars = max_chars or _env_int("UI_DICT_MAX_CHARS", DEFAULT_DICT_MAX_CHARS)
    if len(text) > max_chars:
        raise DictionaryTooLargeError(
            f"This dictionary has {len(text):,} characters; the limit is {max_chars:,}. "
            "Split it into smaller files and upload them separately."
        )
    return text


_DATE_KEYWORD = re.compile(r"date|Date|DATE")
_RANGE_KEYWORD = re.compile(r"range|Range|between")
_WORD = re.compile(r"\w+")
_DIGITS = re.compile(r"\d{1,15}")
_RANGE_SEP = re.compile(r"to|and|-|–")


def extract_pdf_manual_rules(page_text: str) -> Dict[str, Dict[str, Any]]:
    """Best-effort "manual" rule extraction from dictionary PDF text.

    Replaces two regexes of the form `([\\w_]+).*?keyword.*?...`, which took
    cubic time on a long run of word characters (ReDoS, B-03). This version is
    linear: it works line by line, caps each line at _LINE_CAP characters, and
    uses only single left-to-right scans (no nested quantifiers, no
    backtracking). Rules: a line containing a date keyword makes its first word
    a date field; a line with range/between followed by two numbers separated
    by to/and/- makes its first word a numeric range field.
    """
    rules: Dict[str, Dict[str, Any]] = {}
    for line in page_text.splitlines():
        line = line[:_LINE_CAP]
        first = _WORD.search(line)
        if not first:
            continue
        field = first.group(0)
        if field in rules:
            continue
        if _DATE_KEYWORD.search(line, first.start() + 1):
            rules[field] = {"type": "date"}
            continue
        kw = _RANGE_KEYWORD.search(line, first.start() + 1)
        if kw:
            n1 = _DIGITS.search(line, kw.end())
            sep = _RANGE_SEP.search(line, n1.end()) if n1 else None
            n2 = _DIGITS.search(line, sep.end()) if sep else None
            if n1 and n2:
                rules[field] = {"min": int(n1.group(0)), "max": int(n2.group(0))}
    return rules


def validate_dictionary_shape(obj) -> dict:
    """Shape check for a user-supplied JSON dictionary (B-10): a JSON object,
    `rules` (if present) an object, `fields` (if present) a list of objects.
    Raises ValueError with a user-safe message; returns `obj` unchanged."""
    if not isinstance(obj, dict):
        raise ValueError("The dictionary must be a JSON object.")
    if "rules" in obj and not isinstance(obj["rules"], dict):
        raise ValueError("The dictionary's 'rules' entry must be a JSON object.")
    if "fields" in obj:
        fields = obj["fields"]
        if not isinstance(fields, list) or not all(isinstance(f, dict) for f in fields):
            raise ValueError("The dictionary's 'fields' entry must be a list of JSON objects.")
    return obj


def field_markdown(field: dict, detailed: bool = False, max_values: int = 5) -> str:
    """Markdown for one extracted dictionary field with EVERY dictionary/LLM
    derived string escaped (rendered as text, never as markdown/HTML)."""
    name = md_escape(field.get("field_name", "(unnamed)"))
    info = f"**{name}** ({md_escape(field.get('data_type', 'unknown'))})"
    if field.get("required"):
        info += " *[Required]*"
    desc = field.get("description")
    if desc:
        desc = str(desc)
        if detailed and len(desc) > 150:
            desc = desc[:150] + "..."
        info += f"\n   {'📝 ' if detailed else ''}{md_escape(desc)}"
    lo, hi = field.get("min_value"), field.get("max_value")
    if (lo is not None and lo != "") or (hi is not None and hi != ""):
        info += f"\n   {'📊 ' if detailed else ''}Range: {md_escape(lo if lo is not None else 'N/A')} - {md_escape(hi if hi is not None else 'N/A')}"
    allowed = field.get("allowed_values")
    if isinstance(allowed, (list, tuple)) and allowed:
        vals = ", ".join(md_escape(v) for v in allowed[:max_values])
        if len(allowed) > max_values:
            vals += f" ... +{len(allowed) - max_values} more"
        info += f"\n   {'✓ Allowed' if detailed else 'Values'}: {vals}"
    return info


# --- Per-session dictionary cache (B-12) ------------------------------------
_SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def default_cache_base() -> Path:
    """Root of the dictionary disk cache. DICT_CACHE_DIR overrides it. Never
    the legacy shared per-user folder in the home directory (its entries
    must not be readable by new sessions)."""
    override = (os.environ.get("DICT_CACHE_DIR") or "").strip()
    return Path(override) if override else Path(tempfile.gettempdir()) / "data_analyzer_ui_cache_v2"


class DictionaryCache:
    """Disk cache of parsed dictionaries, private to ONE browser session.

    Entries live in `<base>/<session_id>/<sha256>.json` (random 128-bit session
    id held in st.session_state), so a different session can neither read,
    count nor delete them. Keys are SHA-256 of the uploaded bytes. Entries
    expire after `ttl_seconds` (UI_CACHE_TTL_HOURS, default 12) and expired
    entries of every session are pruned opportunistically. JSON only.
    """

    def __init__(self, base_dir: Path, session_id: str, ttl_seconds: Optional[float] = None):
        if not _SESSION_ID_RE.match(session_id or ""):
            raise ValueError("invalid cache session id")
        self.base = Path(base_dir)
        self.session_id = session_id
        self.ttl = ttl_seconds if ttl_seconds is not None else _env_int(
            "UI_CACHE_TTL_HOURS", DEFAULT_CACHE_TTL_HOURS) * 3600
        self.dir = self.base / session_id
        if not self._prepare_dirs():
            # The configured location is not a private directory we own (e.g. a
            # pre-created or symlinked folder in the shared temp dir): use a fresh
            # private one for this process instead.
            self.base = self._fallback_base()
            self.dir = self.base / session_id
            self.dir.mkdir(exist_ok=True, mode=0o700)

    # One private fallback directory per process, created on first need.
    _fallback_dir: Optional[Path] = None

    @staticmethod
    def _is_private_dir(path) -> bool:
        """A real directory (not a symlink), owned by us, with no group/other bits."""
        try:
            st = os.lstat(path)
        except OSError:
            return False
        if not stat_module.S_ISDIR(st.st_mode) or (st.st_mode & 0o077):
            return False
        getuid = getattr(os, "getuid", None)
        return getuid is None or st.st_uid == getuid()

    def _prepare_dirs(self) -> bool:
        try:
            self.base.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not self._is_private_dir(self.base):
                raise OSError("cache base is not a private directory")
            self.dir.mkdir(exist_ok=True, mode=0o700)
            if not self._is_private_dir(self.dir):
                raise OSError("cache session directory is not a private directory")
            return True
        except OSError:
            return False

    @classmethod
    def _fallback_base(cls) -> Path:
        if cls._fallback_dir is None or not cls._is_private_dir(cls._fallback_dir):
            cls._fallback_dir = Path(tempfile.mkdtemp(prefix="data_analyzer_ui_cache_"))
            logger.warning("dictionary_cache_base_unsafe: using a private per-process cache directory instead")
        return cls._fallback_dir

    @staticmethod
    def _open_regular(path: Path, flags: int, mode: int = 0o600) -> int:
        """os.open that refuses symlinks and anything that is not a regular file."""
        # O_NONBLOCK: opening a FIFO planted as an entry must fail, not hang.
        fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0), mode)
        try:
            if not stat_module.S_ISREG(os.fstat(fd).st_mode):
                raise OSError("cache entry is not a regular file")
        except BaseException:
            os.close(fd)
            raise
        return fd

    @staticmethod
    def key_for(content: bytes, llm: bool = False) -> str:
        digest = hashlib.sha256(content).hexdigest()
        return f"{digest}_llm" if llm else digest

    def _path(self, key: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}(_llm)?", key):
            raise ValueError("invalid cache key")
        return self.dir / f"{key}.json"

    def _expired(self, path: Path, now: Optional[float] = None) -> bool:
        try:
            return ((now or time.time()) - path.stat().st_mtime) > self.ttl
        except OSError:
            return True

    def get(self, key: str) -> Optional[dict]:
        path = self._path(key)
        if path.is_symlink():
            path.unlink(missing_ok=True)  # removes the link only, never its target
            return None
        if not path.exists():
            return None
        if self._expired(path):
            path.unlink(missing_ok=True)
            return None
        try:
            fd = self._open_regular(path, os.O_RDONLY)
            with os.fdopen(fd, "r", encoding="utf-8") as f:
                value = json.load(f)
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def put(self, key: str, value: dict) -> None:
        path = self._path(key)
        self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.is_symlink():
            path.unlink(missing_ok=True)
        try:
            fd = self._open_regular(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        except OSError:
            logger.warning("dictionary_cache_entry_refused")
            return
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f)
        self.prune()

    def count(self) -> int:
        """Number of live entries in THIS session only."""
        return sum(1 for p in self.dir.glob("*.json") if not p.is_symlink() and not self._expired(p))

    def clear(self) -> int:
        """Delete this session's entries only; returns how many were removed."""
        removed = 0
        for p in self.dir.glob("*.json"):
            p.unlink(missing_ok=True)
            removed += 1
        return removed

    def prune(self, now: Optional[float] = None) -> int:
        """Delete expired entries (and empty session folders) under the cache
        root, for every session. Returns the number of entries removed."""
        removed = 0
        try:
            for sdir in self.base.iterdir():
                if not (sdir.is_dir() and _SESSION_ID_RE.match(sdir.name)):
                    continue
                for p in sdir.glob("*.json"):
                    if self._expired(p, now):
                        p.unlink(missing_ok=True)
                        removed += 1
                if sdir != self.dir:
                    try:
                        sdir.rmdir()  # only succeeds when empty
                    except OSError:
                        pass
        except OSError:
            logger.warning("dictionary_cache_prune_failed", exc_info=True)
        return removed


def resolve_effective_dictionary(
    demo_dict_selection: str,
    demo_dictionary_built: Optional[dict],
    dict_file_uploaded: bool,
    previous_session_dictionary: Optional[dict],
) -> Optional[dict]:
    """Decide what `st.session_state.dictionary` should be for this render pass.

    Pure function (no Streamlit calls, no CSV parsing) so it can be unit
    tested without a running Streamlit runtime. Centralizes the "which
    dictionary wins" decision that used to be implicit in the demo-dictionary
    selectbox's `if demo_dict != "None":` branch having no `else`. That
    missing `else` was the root cause of the reported bug: selecting a demo
    dictionary, analyzing, then resetting the selector back to "None" left
    the previously-loaded demo dictionary's rules sitting in
    `st.session_state.dictionary` forever, silently driving subsequent
    "Analyze Data Quality" runs.

    Precedence, evaluated in order:
    1. If a dictionary file is currently present in the file uploader
       (`dict_file_uploaded`), the uploaded dictionary wins and is preserved
       as-is (`previous_session_dictionary`) - the upload-handling code
       earlier in the same render pass is the one that actually parses the
       file and assigns `st.session_state.dictionary`, so by the time this
       function runs, `previous_session_dictionary` already reflects that
       upload. This prevents a leftover "None" demo-selector value (e.g.
       from before a file was uploaded) from clobbering an active upload.
    2. Else, if a demo dictionary is selected (`demo_dict_selection` is not
       "None"/falsy), use the freshly-built dict for that selection
       (`demo_dictionary_built`). Because this is the dict the caller just
       built fresh from the CURRENTLY selected demo entry (not whatever was
       previously in session state), switching from one demo dictionary to
       another fully replaces the old one - there is no merge/append with
       the prior selection's rules.
    3. Else (no upload, demo selector is "None"): return `None`, clearing
       any stale dictionary left over from a prior demo selection or a
       dictionary file that has since been removed from the uploader.

    Args:
        demo_dict_selection: current value of the "Or load demo dictionary:"
            selectbox, e.g. "None" or a key from DEMO_DICTIONARIES.
        demo_dictionary_built: the `{"source", "filename", "rules"}` dict
            freshly parsed this render pass for `demo_dict_selection`, or
            None if `demo_dict_selection == "None"`.
        dict_file_uploaded: True if a dictionary file is currently present
            in the dictionary file uploader widget this render pass (i.e.
            `dict_file is not None`).
        previous_session_dictionary: `st.session_state.dictionary` as it was
            before this render pass's decision. Only consulted when
            `dict_file_uploaded` is True, to preserve what the upload
            handling code already assigned.

    Returns:
        The dict that `st.session_state.dictionary` should be set to, or
        None if no dictionary should be active.
    """
    if dict_file_uploaded:
        return previous_session_dictionary

    if demo_dict_selection and demo_dict_selection != "None":
        return demo_dictionary_built

    return None


# Configure Streamlit page
st.set_page_config(
    page_title="Data Quality Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"  # Keep sidebar collapsed by default
)

# Initialize environment banner (bottom position to avoid navbar conflict)
if ENVBANNER_AVAILABLE:
    # Get environment and create custom message with PHI warning
    import os
    app_env = os.getenv("APP_ENV", "dev").lower()
    # Only show banner in dev/staging, hide in production
    if app_env != "prod":
        banner_text = f"{app_env.upper()} - do NOT use real data or files with PHI."
        envbanner.streamlit(position="bottom", opacity=0.9, text=banner_text)

# --- Design system -----------------------------------------------------
#
# Palette (single accent, navy anchor):
#   Navy (navbar / dark text):     #1e293b
#   Navy, deeper (dark mode bg):   #0f172a
#   Slate borders/surfaces:        #e2e8f0 / #334155
#   Accent (primary actions):      #2563eb   (hover/active: #1d4ed8)
#   Accent, soft surfaces:         #eff6ff / #3b82f6 (used sparingly - focus
#                                   rings, links, chart highlight - not a
#                                   second competing hue)
#   Body text:                     #1e293b (light) / #e2e8f0 (dark)
#   Muted text:                    #64748b (light) / #94a3b8 (dark)
#
# NAVBAR APPROACH (version-robust, and actually contains the nav):
# Older versions of this file rendered the title as raw injected HTML
# (`.app-title-bar`, position:fixed) sitting ABOVE st.tabs, with a *separate*
# st.toggle placed in normal document flow just below it - so the "navbar"
# never actually contained navigation or the toggle, just the title.
#
# This version uses `st.container(key="navbar")`, which Streamlit guarantees
# renders a DOM node carrying a stable, public CSS class `.st-key-navbar`
# (documented, public `key=` behavior - unlike `data-baseweb` internals,
# which are private implementation detail and have broken across Streamlit
# versions before, e.g. 1.28 -> 1.59). Real Streamlit widgets (title
# markdown, in-bar navigation, dark-mode toggle) are mounted inside that
# container via ordinary st.* calls - widgets cannot be mounted inside raw
# `unsafe_allow_html` markup, only inside a real container - and the
# container itself is pinned with position:fixed. Navigation inside the bar
# uses st.segmented_control (its own public testid root,
# [data-testid="stSegmentedControl"], not a data-baseweb internal) instead of
# st.tabs, because st.tabs cannot be relocated into an arbitrary container.
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Hide sidebar by default */
    section[data-testid="stSidebar"] {
        display: none !important;
    }

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Leave room for the fixed navbar below; generous, calm spacing */
    .block-container {
        padding-top: 4.75rem !important;
        padding-bottom: 3rem !important;
        max-width: 100% !important;
    }

    .stApp {
        background: #ffffff;
    }

    h1, h2, h3, h4 {
        color: #1e293b;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    h3 {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    p, span, label, li {
        color: #334155;
    }

    /* ---- Fixed navbar: st.container(key="navbar") -> .st-key-navbar ---- */
    .st-key-navbar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        background-color: #1e293b;
        padding: 0.65rem 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.15);
    }

    /* Title + version subscript (plain markdown mounted inside the bar) */
    .app-title {
        color: #f1f5f9;
        font-size: 1.15rem;
        font-weight: 600;
        line-height: 2.2rem;
        white-space: nowrap;
    }
    .app-version {
        color: #94a3b8;
        font-size: 0.68rem;
        font-weight: 500;
        margin-left: 0.4rem;
        vertical-align: sub;
    }

    /* Segmented-control navigation, restyled as light pills on the dark bar.
       Selector targets the widget's own public testid root only - no
       data-baseweb descendant selectors. */
    .st-key-navbar [data-testid="stSegmentedControl"] {
        display: flex;
        justify-content: center;
    }
    .st-key-navbar [data-testid="stSegmentedControl"] label {
        background-color: transparent !important;
        color: #cbd5e1 !important;
        border: none !important;
        font-weight: 500;
    }
    .st-key-navbar [data-testid="stSegmentedControl"] label:hover {
        background-color: #334155 !important;
    }
    .st-key-navbar [data-testid="stSegmentedControl"] label[aria-checked="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
    }

    /* Dark-mode toggle mounted in the bar */
    .st-key-navbar [data-testid="stToggle"] label p {
        color: #cbd5e1 !important;
        font-size: 0.85rem;
    }
    .st-key-navbar div[data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }
    /* Keep every direct widget wrapper inside the bar vertically centered
       and free of the default block spacing so title/nav/toggle line up. */
    .st-key-navbar [data-testid="stVerticalBlockBorderWrapper"],
    .st-key-navbar [data-testid="element-container"] {
        display: flex;
        align-items: center;
    }

    /* ---- Cards: shared surface treatment for upload / dictionary / results ----
       Applied via st.container(key="card-...") -> .st-key-card-... */
    [class*="st-key-card-"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.25rem 1.5rem 1.5rem 1.5rem;
        margin-bottom: 0.75rem;
    }
    .app-section-title {
        margin-top: 0 !important;
        font-size: 1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.75rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #ffffff;
        border: 1.5px dashed #cbd5e1;
        border-radius: 8px;
        transition: border-color 0.15s ease, background-color 0.15s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #2563eb;
        background: #eff6ff;
    }

    /* Primary button: well-proportioned, restrained (native button shape,
       one accent color, no gradient/glow). */
    .stButton > button[kind="primary"] {
        background-color: #2563eb;
        border: none;
        font-weight: 600;
        transition: background-color 0.15s ease;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8;
    }
    .stButton > button[kind="secondary"] {
        font-weight: 500;
    }

    /* Metrics: quiet surface, accent reserved for the number itself */
    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 0.9rem 1rem;
        border-radius: 8px;
    }
    [data-testid="stMetricValue"] {
        color: #1e293b;
    }

    [data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }

    .dataframe {
        font-size: 0.9rem;
    }
    .element-container {
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Navbar: title + in-bar navigation + dark-mode toggle, all inside one
# real st.container(key=...) so they are visually and structurally contained
# by the fixed bar (not stacked in normal flow beneath it). ------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

with st.container(key="navbar"):
    _nav_title_col, _nav_links_col, _nav_toggle_col = st.columns([2.2, 2.5, 1])

    with _nav_title_col:
        st.markdown(
            f'<span class="app-title">Data Quality Analyzer'
            f'<sub class="app-version">v{__version__}</sub></span>',
            unsafe_allow_html=True,
        )

    with _nav_links_col:
        nav_choice = st.segmented_control(
            "Navigation",
            ["Analyze", "About"],
            default="Analyze",
            required=True,
            key="nav_choice",
            label_visibility="collapsed",
        )

    with _nav_toggle_col:
        st.toggle("🌙", key="dark_mode", help="Toggle dark mode")

# segmented_control returns None if a user deselects the active pill by
# clicking it again - treat that the same as staying on "Analyze" so the app
# never renders a blank page.
nav_choice = nav_choice or "Analyze"

# --- Dark mode -----------------------------------------------------------
# Coherent restyle (not a bolted-on override): same card/spacing/radius
# system as light mode, just remapped color tokens. Persisted via
# st.session_state so switching pages or re-running the script doesn't reset
# it, and toggling itself never touches st.session_state.data/dictionary/
# analysis_results, so uploaded data and results survive a theme change.
if st.session_state.dark_mode:
    st.markdown("""
    <style>
        .stApp, .block-container {
            background: #0f172a !important;
        }
        .stApp, .block-container, p, span, label, li,
        div[data-testid="stMarkdownContainer"] {
            color: #e2e8f0 !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #f1f5f9 !important;
        }
        [class*="st-key-card-"] {
            background: #1e293b !important;
            border-color: #334155 !important;
        }
        .app-section-title {
            color: #f1f5f9 !important;
        }
        [data-testid="stExpander"] {
            background: #1e293b !important;
            border: 1px solid #334155 !important;
        }
        [data-testid="stExpander"] summary {
            background: #1e293b !important;
            color: #e2e8f0 !important;
        }
        input, textarea, select,
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            background-color: #1e293b !important;
            color: #e2e8f0 !important;
            border-color: #334155 !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: #1e293b !important;
            border-color: #334155 !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: #3b82f6 !important;
            background: #1e2a3f !important;
        }
        pre, code {
            background-color: #1e293b !important;
            color: #e2e8f0 !important;
        }
        [data-testid="stMetric"] {
            background: #1e293b !important;
            border: 1px solid #334155 !important;
        }
        [data-testid="stMetricValue"] {
            color: #f1f5f9 !important;
        }
        [data-testid="stDataFrame"] {
            background: #1e293b !important;
        }
        .stButton > button[kind="secondary"] {
            background-color: #1e293b !important;
            color: #e2e8f0 !important;
            border-color: #334155 !important;
        }
    </style>
    """, unsafe_allow_html=True)
# --- End design system -----------------------------------------------------

class DataQualityAnalyzer:
    """
    HTTP client for the centralized data-analyzer REST API.

    Historically this class ran mcp_server.QualityPipeline in-process. It now
    POSTs to the API's POST /api/v1/analyze endpoint instead, so there is a
    single running backend for rule-based quality checks shared by the API,
    the MCP server, and this UI (Option A: centralize the runtime).

    Conditional logic validation (extracted from a dictionary's 'fields' via
    RuleExtractor/LogicValidator) is NOT part of /api/v1/analyze - that
    endpoint is intentionally rule-based only (schema/range/allowed-value
    checks, no LLM). Logic validation stays local here and its violations are
    merged into the API's issues/summary after the HTTP call returns.

    Configuration (environment variables):
        DATA_ANALYZER_API_URL: Base URL of the data-analyzer REST API.
            Defaults to "http://localhost:8000" ONLY for local dev (APP_ENV
            dev/local/test and not on Azure Container Apps). In a deployed
            environment there is no fallback: an unset value is a config
            error (self.config_error), shown to the user as one line, and
            entrypoint.sh refuses to start the container at all.
        DATA_ANALYZER_API_KEY: Value sent as the X-API-Key header. Must match
            the API app's own DATA_ANALYZER_API_KEY.

    Cold start: the API scales to zero, and its first request after idle can
    take ~45 s. Each attempt waits ANALYZE_TIMEOUT_SECONDS, and a connection
    error / timeout / gateway 502-504 is retried ONCE after notifying the
    caller via `on_status` ("starting the analysis service...").
    """

    ANALYZE_TIMEOUT_SECONDS = 90
    COLD_START_RETRY_DELAY_SECONDS = 5
    _RETRYABLE_STATUS = frozenset({502, 503, 504})
    STARTING_MESSAGE = (
        "Starting the analysis service (it sleeps when idle; this can take up to a minute)... retrying."
    )

    def __init__(self):
        self.config_error: Optional[str] = None
        try:
            self.api_base_url: Optional[str] = resolve_api_base_url()
        except ApiConfigError as e:
            self.api_base_url = None
            self.config_error = str(e)
            logger.error("ui_config_error setting=DATA_ANALYZER_API_URL message=%r", self.config_error)
        self.api_key = os.getenv("DATA_ANALYZER_API_KEY", "")
        # Optional callable(str) for user-visible progress (e.g. st.info).
        self.on_status = None

    async def analyze_data_quality(self, data: pd.DataFrame, dictionary: Optional[Dict] = None, source_format: Optional[str] = None) -> Dict[str, Any]:
        """
        Run data quality analysis via the centralized REST API.

        Args:
            data: DataFrame to analyze
            dictionary: Optional dictionary with validation rules and schema.
                       Can have 'rules' and/or 'schema' keys, or direct field definitions.
            source_format: Optional hint for how to serialize `data` for the
                       HTTP upload. One of "xlsx", "xls", or None (default:
                       CSV). When the original upload was an Excel file, pass
                       "xlsx"/"xls" so the API receives real Excel bytes with
                       the correct content-type instead of a CSV re-encoding.

        Returns:
            Dict with summary, issues, recommendations, quality_checks, and
            summary_stats - the exact shape the rest of this module (the
            dashboard rendering code) has always consumed.

        Raises:
            RuntimeError: if the API call fails (non-200 response, connection
                error, or timeout). Callers should catch this and surface it
                via st.error(...) rather than letting the UI crash.
        """
        # Parse dictionary to extract schema and rules to send to the API
        # (unchanged from the old in-process code path - only the destination
        # of schema/rules changed, from a local QualityPipeline call to an
        # HTTP request).
        schema = None
        rules = None
        # Columns whose dictionary type has no checkable equivalent: NOT sent as
        # schema and NOT a failure - reported on the page and in the exports.
        types_not_checked: list = []

        if dictionary and isinstance(dictionary, dict):
            # Handle different dictionary formats
            if 'rules' in dictionary:
                dict_rules = dictionary['rules']
                # Convert web_app dictionary format to QualityChecker format
                schema = {}
                rules = {}
                for field_name, field_spec in dict_rules.items():
                    if isinstance(field_spec, dict):
                        # Extract type for schema
                        if 'type' in field_spec:
                            canon = normalize_schema_type(field_spec['type'])
                            if canon is None:
                                types_not_checked.append(str(field_name))
                            else:
                                schema[field_name] = canon

                        # Extract validation rules
                        field_rules = {}
                        if 'min' in field_spec and pd.notna(field_spec['min']):
                            try:
                                field_rules['min'] = float(field_spec['min'])
                            except (ValueError, TypeError):
                                pass
                        if 'max' in field_spec and pd.notna(field_spec['max']):
                            try:
                                field_rules['max'] = float(field_spec['max'])
                            except (ValueError, TypeError):
                                pass
                        if 'allowed_values' in field_spec and field_spec['allowed_values']:
                            field_rules['allowed'] = field_spec['allowed_values']

                        if field_rules:
                            rules[field_name] = field_rules

            elif 'schema' in dictionary:
                raw_schema = dictionary.get('schema')
                if isinstance(raw_schema, dict):
                    schema, types_not_checked = normalize_schema(raw_schema)
                else:
                    schema = raw_schema
                rules = dictionary.get('validation_rules', {})

        # Run the rule-based engine via the API (replaces the old in-process
        # QualityPipeline(data, schema, rules).run_all_checks() call). The
        # API already returns issues/summary/recommendations/quality_checks/
        # summary_stats in this exact shape - no reshaping needed here.
        result = self._call_analyze_api(data, schema, rules, source_format=source_format)

        issues = result["issues"]
        summary = result["summary"]
        recommendations = result["recommendations"]

        # LOGIC VALIDATION - Run conditional logic checks if available.
        # A failure here is NOT silent: it is logged and returned as
        # `logic_validation_error`, which the dashboard shows as a warning
        # (the rule-based results are still valid and shown).
        logic_validation_error = None
        # This is separate from the rule-based REST engine (it's driven by
        # dictionary 'fields' extracted via RuleExtractor, not schema/rules),
        # so it stays local and its results are merged in after the API call.
        logic_violations_count = 0
        rule_coverage = None  # RuleCoverage.to_dict() once conditional rules were checked
        if PROGRAM_CACHE_AVAILABLE and dictionary and 'fields' in dictionary:
            try:
                # Extract conditional rules from dictionary fields
                extractor = RuleExtractor()
                conditional_rules = extractor.extract_rules_from_fields(
                    dictionary.get('fields', []),
                    format_type="REDCap"  # Default format, could be configurable
                )

                if conditional_rules:
                    # Run logic validation
                    validator = LogicValidator()
                    logic_violations, coverage = validator.validate_with_coverage(conditional_rules, data)
                    rule_coverage = coverage.to_dict()
                    if not coverage.complete:
                        # Never let an unchecked rule look like a clean one.
                        logger.warning("logic_rule_coverage_incomplete %s", coverage.summary())

                    # Add logic violations to issues list
                    for violation in logic_violations:
                        issues.append({
                            "type": "logic_violation",
                            "severity": violation.severity,
                            "column": ', '.join(violation.affected_fields),
                            "row": violation.row_index,
                            "value": violation.actual_values,
                            "message": f"Logic rule violated: {violation.rule_description}"
                        })

                    logic_violations_count = len(logic_violations)
            except Exception as e:
                logic_validation_error = (
                    f"Conditional-logic checks could not run ({type(e).__name__}: {e}). "
                    "The results below cover the rule-based checks only."
                )
                logger.error("logic_validation_failed error_class=%s", type(e).__name__, exc_info=True)

        if logic_violations_count:
            # Logic issues were appended after the API computed its counts -
            # recompute them so the summary stays consistent with `issues`.
            # Add to the API's true totals (its list may be shortened), never
            # recompute from the listed issues.
            summary["issues_found"] = summary.get("issues_found", 0) + logic_violations_count
            summary["critical_issues"] = summary.get("critical_issues", 0) + sum(
                1 for v in logic_violations if v.severity == 'error')
            summary["warnings"] = summary.get("warnings", 0) + sum(
                1 for v in logic_violations if v.severity == 'warning')
            recommendations.append({
                "type": "conditional_logic",
                "priority": "critical",
                "message": "Conditional logic violations detected. Review field dependencies and branching rules in data dictionary"
            })
        summary["logic_violations_count"] = logic_violations_count
        # True per-type counts (the API lists only the first rows of each finding).
        summary["issue_type_totals"] = summary.get("issue_type_totals") or {}
        summary["rules_not_checked"] = rule_coverage["rules_not_checked"] if rule_coverage else 0
        summary["column_types_not_checked"] = list(types_not_checked)  # names; count = len()

        return {
            "summary": summary,
            "issues": issues,
            "recommendations": recommendations,
            "quality_checks": result["quality_checks"],
            "summary_stats": result["summary_stats"],
            "logic_validation_error": logic_validation_error,
            "rule_coverage": rule_coverage,
        }

    def _notify(self, message: str) -> None:
        if self.on_status:
            try:
                self.on_status(message)
            except Exception:  # a UI callback must never break the request
                logger.warning("on_status callback failed", exc_info=True)

    # Content-types matching api_server.py's /api/v1/analyze extension
    # dispatch (and the standard IANA media types for each Excel format).
    _EXCEL_CONTENT_TYPES = {
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xls": "application/vnd.ms-excel",
    }

    def _call_analyze_api(self, data: pd.DataFrame, schema: Optional[Dict], rules: Optional[Dict], source_format: Optional[str] = None) -> Dict[str, Any]:
        """POST a DataFrame to the API's /api/v1/analyze endpoint and return
        the parsed JSON report.

        Args:
            data: DataFrame to analyze.
            schema: Optional column-type map, sent as a JSON form field.
            rules: Optional validation-rule map, sent as a JSON form field.
            source_format: "xlsx", "xls", or None. When set, `data` is
                re-encoded as a real Excel workbook and uploaded with the
                matching Excel content-type so the API's Excel branch
                (mcp_server.DataLoader.load_excel) handles it. Defaults to
                CSV serialization, which the API also fully supports.

        Returns:
            The parsed JSON body: {"summary", "issues", "recommendations",
            "quality_checks", "summary_stats"}.

        Raises:
            RuntimeError: on any non-200 response, connection error, or
                timeout, with a message suitable for st.error(...).
        """
        source_format = (source_format or "").lower().lstrip(".") or None

        if source_format in self._EXCEL_CONTENT_TYPES:
            excel_buffer = io.BytesIO()
            data.to_excel(excel_buffer, index=False, engine="openpyxl")
            file_bytes = excel_buffer.getvalue()
            filename = f"data.{source_format}"
            content_type = self._EXCEL_CONTENT_TYPES[source_format]
        else:
            file_bytes = data.to_csv(index=False).encode("utf-8")
            filename = "data.csv"
            content_type = "text/csv"

        files = {"data_file": (filename, file_bytes, content_type)}

        form_data = {}
        if schema:
            form_data["schema"] = json.dumps(schema, default=str)
        if rules:
            form_data["rules"] = json.dumps(rules, default=str)

        if self.config_error or not self.api_base_url:
            raise ApiConfigError(self.config_error or "Configuration error: no data-analyzer API URL.")

        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        url = f"{self.api_base_url}/api/v1/analyze"
        timeout = self.ANALYZE_TIMEOUT_SECONDS

        response = None
        for attempt in (1, 2):
            failure = None
            try:
                # allow_redirects=False: a redirect would re-send the X-API-Key to wherever
                # the Location header points; any 3xx is treated as an error below.
                response = requests.post(url, files=files, data=form_data, headers=headers,
                                         timeout=timeout, allow_redirects=False)
            except requests.exceptions.ConnectionError as e:
                failure = (f"Could not connect to the analysis service at {self.api_base_url} "
                           f"({type(e).__name__}).", type(e).__name__, None)
            except requests.exceptions.Timeout as e:
                failure = (f"The analysis service timed out (no response within {timeout}s).", type(e).__name__, None)
            except requests.exceptions.RequestException as e:
                logger.error("api_call_failed url=%s error_class=%s", url, type(e).__name__)
                raise RuntimeError(f"Data-analyzer API request failed: {type(e).__name__}: {e}")
            else:
                if response.status_code in self._RETRYABLE_STATUS:
                    failure = (f"The analysis service is unavailable (HTTP {response.status_code}).",
                               "http_error", response.status_code)

            if failure is None:
                break
            message, error_class, http_status = failure
            logger.error("api_call_failed url=%s attempt=%d error_class=%s http_status=%s",
                         url, attempt, error_class, http_status)
            if attempt == 1:
                self._notify(self.STARTING_MESSAGE)
                time.sleep(self.COLD_START_RETRY_DELAY_SECONDS)
                continue
            raise RuntimeError(f"{message} Retried once; please try again in a minute or contact the app owner.")

        if 300 <= response.status_code < 400:
            logger.error("api_call_failed url=%s error_class=redirect_refused http_status=%s",
                         url, response.status_code)
            raise RuntimeError(
                f"Data-analyzer API answered with a redirect (HTTP {response.status_code}), "
                "which the UI refuses to follow because the request carries the API key. "
                "Check that DATA_ANALYZER_API_URL is the API's final https URL.")

        if response.status_code != 200:
            try:
                error_body = response.json()
                detail = error_body.get("detail") or error_body.get("error") or response.text
            except ValueError:
                detail = response.text
            logger.error("api_call_failed url=%s error_class=http_error http_status=%s", url, response.status_code)
            raise RuntimeError(f"Data-analyzer API returned HTTP {response.status_code}: {detail}")

        try:
            return response.json()
        except ValueError as e:
            raise RuntimeError(f"Data-analyzer API returned a non-JSON response: {e}")

def load_demo_data(dataset_name: str):
    """Load demo dataset matching the dictionary options"""
    demo_data = {
        'western': pd.DataFrame({
            'employee_id': [1001, 1002, 1003, 1004, 1005],
            'first_name': ['John', 'Jane', 'Mike', 'Bob', 'Alice'],
            'last_name': ['Smith', 'Doe', 'Johnson', 'Brown', 'Wilson'],
            'age': [35, 28, 67, 45, 32],  # ERROR: 67 is outside range (max 65)
            'salary': [75000, 85000, 45000, 95000, None],  # ERROR: 45000 below min (50000), WARNING: None is missing
            'hire_date': ['2022-03-15', '2023-01-10', '2023-99-99', '2021-07-22', '2022-11-30'],  # ERROR: invalid date 2023-99-99
            'last_login_datetime': ['2023-01-15 10:30:00', '2023-02-20 14:45:00', '2023-03-10 09:15:00', '2023-04-05 16:20:00', '2023-05-12 11:00:00'],
            'bonus_percentage': [5.5, 10.0, 15.5, 8.0, 12.5],
            'department': ['Engineering', 'Marketing', 'InvalidDept', 'Sales', 'Finance'],  # ERROR: InvalidDept not in allowed values
            'is_active': [True, True, False, None, True],  # WARNING: None is missing
            'skills': ['Python;SQL', 'Marketing;Analytics', 'Java;AWS', 'Sales;CRM', 'Python;Leadership'],
            'email': ['john@company.com', 'jane@company.com', 'mike@company.com', 'bob@company.com', 'alice@company.com'],
            'phone': ['+1-555-1234', '+1-555-5678', '+1-555-9012', None, '+1-555-3456']  # WARNING: None is missing
        }),
        'clinical': pd.DataFrame({
            'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008'],
            'age': [45, 62, 73, 28, 155, 39, 67, 51],  # 155 is invalid age
            'gender': ['M', 'F', 'M', 'F', 'X', 'F', 'M', 'invalid'],  # X and invalid are issues
            'diagnosis_code': ['I21.0', 'J18.9', 'N18.9', 'K92.2', 'invalid', 'G20.9', 'E11.9', ''],
            'admission_date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', 'invalid-date', '2024-01-20', '2024-01-21', '2024-01-22'],
            'discharge_date': ['2024-01-20', '2024-01-22', '2024-01-25', '2024-01-19', None, '2024-01-21', '', '2024-01-23'],
            'treatment_type': ['Emergency', 'Inpatient', 'Inpatient', 'Observation', 'invalid', 'Outpatient', 'Inpatient', 'Unknown'],
            'lab_result_wbc': [7.5, 12.3, 6.8, 8.2, 50.0, 7.8, 10.5, None],  # 50.0 is out of range
            'lab_result_hemoglobin': [14.2, 12.8, 10.5, 13.5, 'invalid', 14.5, None, 13.2],
            'blood_pressure_systolic': [130, 145, 155, 110, 250, 125, 165, 135],  # 250 is too high
            'blood_pressure_diastolic': [85, 92, 95, 70, 130, 80, 98, 82],  # 130 is too high
            'temperature': [37.2, 38.5, 36.8, 36.9, 45.0, 36.7, 37.3, None],  # 45.0 is impossible
            'heart_rate': [88, 96, 78, 72, 200, 75, 90, 68],  # 200 is too high
            'follow_up_required': ['Yes', 'Yes', 'Yes', 'No', 'Maybe', 'Yes', 'Yes', None],  # Maybe is invalid
            'outcome_status': ['Improved', 'Recovered', 'Stable', 'Recovered', 'invalid', 'Stable', 'Ongoing', 'Improved'],
            'length_of_stay': [5, 6, 8, 1, 500, 1, None, 1]  # 500 days is unrealistic
        }),
        'asian': pd.DataFrame({
            'staff_id': [2001, 2002, 2003, 2004, 2005],
            'given_name': ['Akiko', 'Wei', None, 'Raj', 'Mei'],
            'family_name': ['Tanaka', 'Zhang', 'Kumar', 'invalid', 'Chen'],
            'age': [30, 21, 45, 38, 62],  # 21 below min, 62 above max
            'monthly_salary': [8500, 9200, 6000, 10500, None],  # 6000 below min
            'join_date': ['2020-06-01', 'invalid-date', '2021-03-15', '2022-09-10', '2023-01-20'],
            'dept_code': ['DEV', 'MKT', 'invalid', 'OPS', 'FIN'],
            'active_status': [1, 1, 0, None, 1],
            'work_email': ['akiko@work.com', 'wei@work.com', 'invalid', 'raj@work.com', 'mei@work.com']
        }),
        'mixed': pd.DataFrame({
            'id': [3001, 3002, 3003, 3004, 3005],
            'name_first': ['Carlos', 'Emma', 'invalid', 'Liu', None],
            'name_last': ['Rodriguez', 'invalid', 'Brown', 'Wang', 'Lee'],
            'age': [40, 24, 35, 56, 45],  # 24 below min, 56 above max
            'salary': [70000, 80000, 60000, 90000, None],  # 60000 below min, 90000 above max
            'hired': ['2022-05-01', '2023-08-15', 'invalid-date', '2021-12-01', '2023-03-10'],
            'active': [True, False, None, True, True],
            'department': ['Research', 'invalid', 'Engineering', 'Quality', 'Sales']
        }),
        # Synthetic REDCap-style clinical dataset matching the field set of the
        # "REDCap - Clinical (synthetic, with branching logic)" demo dictionary
        # in demo_dictionaries.py (16 columns spanning the demographics,
        # treatment, medical_history, and safety REDCap forms). 20 rows,
        # deliberately containing several quality issues so the demo produces
        # findings:
        #   1. age[4] = 91          -> ERROR: exceeds dictionary max (85)
        #   2. gender[7] = 'Unknown'-> ERROR: not in allowed values (Male/Female/Other)
        #   3. lab_glucose[9] = None-> WARNING: missing value where diabetic + required
        #   4. dosage_mg[9] = 750   -> ERROR: exceeds dictionary max (500)
        #   5. BRANCHING-LOGIC VIOLATION row 14 (subject TEST-014): pregnant='Yes'
        #      but gender='Male' (pregnant should only be answerable when
        #      gender='Female' per the dictionary's branching logic) -
        #      demonstrates a logic-check finding.
        'redcap_clinical': pd.DataFrame({
            'subject_id': [f'TEST-{i:03d}' for i in range(1, 21)],
            'age': [34, 45, 58, 91, 27, 62, 39, 50, 71, 44,  # row 4 (idx3) = 91 -> ERROR (max 85)
                    29, 55, 48, 36, 60, 41, 33, 52, 67, 25],
            'gender': ['Female', 'Male', 'Female', 'Male', 'Other', 'Female', 'Unknown', 'Male', 'Female', 'Male',
                       'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],  # row 7 (idx6) = 'Unknown' -> ERROR
            'pregnant': ['No', None, 'Yes', None, None, 'No', None, None, 'No', None,
                         'Yes', None, 'No', 'Yes', 'No', None, 'No', None, 'No', None],  # row 14 (idx13) = 'Yes' but gender[13]='Male' -> BRANCHING-LOGIC VIOLATION
            'weeks_pregnant': [None, None, 24, None, None, None, None, None, None, None,
                               12, None, None, None, None, None, None, None, None, None],
            'due_date': [None, None, '2024-08-15', None, None, None, None, None, None, None,
                         '2024-11-02', None, None, None, None, None, None, None, None, None],
            'treatment_arm': ['Active Treatment', 'Placebo', 'Active Treatment', 'Placebo', 'Active Treatment',
                               'Placebo', 'Active Treatment', 'Placebo', 'Active Treatment', 'Placebo',
                               'Active Treatment', 'Placebo', 'Active Treatment', 'Placebo', 'Active Treatment',
                               'Placebo', 'Active Treatment', 'Placebo', 'Active Treatment', 'Placebo'],
            'dosage_mg': [120, None, 85, None, 200, None, 150, None, 750, None,  # row 9 (idx8) = 750 -> ERROR (max 500)
                          95, None, 180, None, 110, None, 60, None, 140, None],
            'placebo_type': [None, 'Tablet', None, 'Capsule', None, 'Tablet', None, 'Capsule', None, 'Tablet',
                              None, 'Capsule', None, 'Tablet', None, 'Capsule', None, 'Tablet', None, 'Capsule'],
            'diabetes': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes',
                         'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'],
            'diabetes_type': [None, 'Type 2', None, 'Type 1', None, 'Type 2', None, 'Type 2', None, 'Type 1',
                               None, 'Type 2', None, 'Type 2', None, 'Type 1', None, 'Type 2', None, 'Type 2'],
            'insulin_dependent': [None, 'No', None, 'Yes', None, 'No', None, 'No', None, 'Yes',
                                   None, 'No', None, 'No', None, 'Yes', None, 'No', None, 'No'],
            'lab_glucose': [None, 145.0, None, 210.5, None, 130.0, None, None, None, 190.0,  # row 8 (idx7): diabetic + required but missing -> WARNING
                             None, 128.0, None, 175.0, None, 205.0, None, 122.0, None, 168.0],
            'adverse_event': ['No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'No', 'Yes',
                               'No', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'No', 'No'],
            'ae_description': [None, None, 'Mild headache, resolved', None, None, 'Nausea after dosing', None, None, None, 'Dizziness',
                                None, None, 'Injection site rash', None, None, None, 'Fatigue', None, None, None],
            'ae_severity': [None, None, 'Mild', None, None, 'Moderate', None, None, None, 'Mild',
                             None, None, 'Mild', None, None, None, 'Moderate', None, None, None],
        })
    }
    return demo_data.get(dataset_name, demo_data['western'])

# Severity vocabulary used across the codebase (verified against api_server.py's
# _build_quality_report and src/logic_engine.py's LogicValidator/ConditionalRule):
# only "error" and "warning" are ever emitted. logic_violation issues (merged in
# by DataQualityAnalyzer.analyze_data_quality from `violation.severity`) also draw
# from that same two-value vocabulary - src/logic_engine.py never emits "critical"
# or any other string. We still map defensively: anything that isn't literally
# "warning" is treated as the red/error tier, so an unexpected future severity
# string degrades to "shown as an error" rather than silently vanishing.
_HEATMAP_WARNING_SEVERITIES = {'warning'}


def _build_issue_matrix(df: pd.DataFrame, issues: list, max_display_rows: int = 60, max_display_cols: int = 100):
    """Pure helper: turn a DataFrame + issues list into heatmap inputs.

    Separated from create_issue_heatmap() so the row/column mapping and
    severity classification logic can be unit tested without a Streamlit
    runtime (create_issue_heatmap has no return value since it renders
    directly via st.plotly_chart/st.caption).

    Returns a dict with:
        issue_matrix: np.ndarray (display_rows x display_cols) of 0/1/2
        hover_text: list of lists of hover strings, same shape
        display_rows, display_cols, row_factor, col_factor: layout ints
        unmapped_column_warnings: dict[column_name -> count] of column-level
            issues (e.g. missing_values) that have no 'row' key and so
            cannot be placed on the per-cell grid
    """
    rows, cols = len(df), len(df.columns)

    row_factor = max(1, rows // max_display_rows)
    col_factor = max(1, cols // max_display_cols)

    display_rows = min(rows, max_display_rows)
    display_cols = min(cols, max_display_cols)

    issue_matrix = np.zeros((display_rows, display_cols))
    hover_text = [['' for _ in range(display_cols)] for _ in range(display_rows)]
    unmapped_column_warnings: Dict[str, int] = {}

    for issue in issues:
        if 'row' in issue and 'column' in issue:
            try:
                col_idx = df.columns.get_loc(issue['column'])
                row_idx = issue['row']

                # Map to display coordinates
                display_row = min(row_idx // row_factor, display_rows - 1)
                display_col = min(col_idx // col_factor, display_cols - 1)

                # Set severity (2 for error/other, 1 for warning). Only the
                # "warning" string maps to the yellow tier; everything else
                # (currently just "error") maps to the red tier.
                severity_value = 1 if issue.get('severity') in _HEATMAP_WARNING_SEVERITIES else 2
                issue_matrix[display_row, display_col] = max(issue_matrix[display_row, display_col], severity_value)

                # Build hover text
                issue_info = f"<b>{issue['type'].replace('_', ' ').title()}</b><br>"
                issue_info += f"Row: {row_idx}<br>"
                issue_info += f"Column: {issue['column']}<br>"
                issue_info += f"Value: {issue.get('value', 'N/A')}<br>"
                issue_info += f"Severity: {issue.get('severity', 'unknown')}"

                if hover_text[display_row][display_col]:
                    hover_text[display_row][display_col] += "<br><br>" + issue_info
                else:
                    hover_text[display_row][display_col] = issue_info
            except Exception:
                pass
        else:
            # Column-level aggregate issue (e.g. missing_values warnings built
            # in api_server.py's _build_quality_report) - no row coordinate
            # exists to plot a cell, so track it separately for the caption.
            col_name = issue.get('column')
            if col_name is not None:
                unmapped_column_warnings[col_name] = unmapped_column_warnings.get(col_name, 0) + 1

    return {
        'issue_matrix': issue_matrix,
        'hover_text': hover_text,
        'display_rows': display_rows,
        'display_cols': display_cols,
        'row_factor': row_factor,
        'col_factor': col_factor,
        'unmapped_column_warnings': unmapped_column_warnings,
    }


def create_issue_heatmap(df: pd.DataFrame, issues: list):
    """Create an interactive heatmap with hover tooltips using Plotly"""
    try:
        rows, cols = len(df), len(df.columns)
        max_display_rows = 60
        max_display_cols = 100  # Increased for wide datasets

        built = _build_issue_matrix(df, issues, max_display_rows, max_display_cols)
        issue_matrix = built['issue_matrix']
        hover_text = built['hover_text']
        row_factor = built['row_factor']
        col_factor = built['col_factor']
        unmapped_column_warnings = built['unmapped_column_warnings']

        # Calculate aspect ratio for proper dimensions
        aspect_ratio = cols / rows
        if aspect_ratio > 1:
            # Wide dataset
            fig_width = 300
            fig_height = max(50, 300 / aspect_ratio)
        else:
            # Tall dataset
            fig_height = 300
            fig_width = max(50, 300 * aspect_ratio)

        # Create interactive Plotly heatmap. zmin/zmax are pinned explicitly so
        # the colorscale mapping (white/yellow/red at z=0/1/2) stays stable
        # regardless of which severities are actually present - without this,
        # a dataset with only warning cells (max z=1) would have Plotly
        # normalize 1 -> the top of the scale (red) instead of the middle.
        fig = go.Figure(data=go.Heatmap(
            z=issue_matrix,
            zmin=0,
            zmax=2,
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            colorscale=[
                [0, '#ffffff'],      # White for no issue
                [0.5, '#fbbf24'],    # Yellow for warning
                [1, '#ef4444']       # Red for error
            ],
            showscale=False,
            xgap=1,
            ygap=1
        ))

        # Update layout. All color-related properties are set explicitly
        # (template + paper/plot bgcolor + font/axis colors) so the figure
        # renders identically regardless of Streamlit's active theme or the
        # app's custom dark-mode CSS overrides - without an explicit
        # template, dark-mode CSS was overriding the plot to solid orange
        # with no visible cell boundaries.
        fig.update_layout(
            template='plotly_white',
            title={
                'text': f'{rows} rows × {cols} cols' + (f' (scale {row_factor}:{col_factor})' if rows > max_display_rows or cols > max_display_cols else ''),
                'font': {'size': 10, 'color': '#111111'}
            },
            width=fig_width,
            height=fig_height,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis={'showticklabels': False, 'showgrid': False, 'color': '#111111'},
            yaxis={'showticklabels': False, 'showgrid': False, 'color': '#111111'},
            paper_bgcolor='white',
            plot_bgcolor='white',
            font_color='#111111'
        )

        # Display with Streamlit
        st.plotly_chart(fig, use_container_width=False)

        # Show summary. Cell-level counts come from the plotted matrix;
        # column-level (unmappable) warnings - e.g. missing_values issues,
        # which are column aggregates with no row coordinate - are called
        # out separately so the caption never claims "0 warnings" when
        # warnings exist but simply couldn't be drawn as cells.
        error_count = np.sum(issue_matrix == 2)
        warning_count = np.sum(issue_matrix == 1)
        unmapped_count = sum(unmapped_column_warnings.values())

        caption_parts = []
        if error_count > 0 or warning_count > 0:
            caption_parts.append(f"🔴 {int(error_count)} cells with errors, 🟡 {int(warning_count)} cells with warnings")
        if unmapped_count > 0:
            cols_str = ', '.join(md_escape(c) for c in list(unmapped_column_warnings.keys())[:5])
            if len(unmapped_column_warnings) > 5:
                cols_str += f", +{len(unmapped_column_warnings) - 5} more"
            caption_parts.append(
                f"⚠️ {unmapped_count} additional column-level warning(s) not shown as cells "
                f"(e.g. missing-value summaries for: {cols_str}) — see Issues Found below"
            )
        if caption_parts:
            st.caption(" · ".join(caption_parts))

    except Exception as e:
        st.caption(f"Issue map unavailable ({type(e).__name__})")

def analysis_inputs_fingerprint(data, dictionary) -> Optional[str]:
    """Short fingerprint of the data + dictionary an analysis ran on (shape, column
    names, a sample of rows, and the dictionary content). Used to notice that stored
    results are stale."""
    if data is None:
        return None
    h = hashlib.sha256()
    h.update(repr((data.shape, [str(c) for c in data.columns])).encode())
    sample = pd.concat([data.head(200), data.tail(200)]).astype(str)
    h.update(pd.util.hash_pandas_object(sample, index=True).values.tobytes())
    h.update(json.dumps(dictionary, sort_keys=True, default=str).encode() if dictionary else b"-")
    return h.hexdigest()


EXPORT_FORMATS = ["Excel with highlighting", "JSON report"]


def render_export_controls(data, results: dict) -> None:
    """Export section: a format choice (Excel is the default, so a download button
    is visible immediately) and a download button built from THESE results (issues,
    rule coverage, types not checked, true totals). The file is built only when the
    user clicks (download_button accepts a callable)."""
    st.markdown('<div class="app-section-title">Export</div>', unsafe_allow_html=True)
    export_format = st.selectbox("Choose format:", EXPORT_FORMATS, key="export_format")
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if export_format == "Excel with highlighting":
        st.download_button(
            label="📊 Download Excel",
            data=lambda: export_to_excel_with_highlighting(
                data, results['issues'], results.get('rule_coverage'),
                results['summary'].get('column_types_not_checked')),
            file_name=f"data_quality_{stamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True, type="secondary")
    else:
        st.download_button(
            label="📄 Download JSON",
            data=lambda: json.dumps(results, indent=2, default=str),
            file_name=f"quality_report_{stamp}.json",
            mime="application/json",
            use_container_width=True, type="secondary")


def export_to_excel_with_highlighting(df: pd.DataFrame, issues: list, rule_coverage: Optional[dict] = None,
                                      types_not_checked: Optional[list] = None) -> bytes:
    """Export data to Excel with error cells highlighted.

    Text that starts with = + - @ (also after whitespace/tab/CR) - in data
    cells, column headers and issue text - is stored as a literal text cell,
    never a live formula (CWE-1236). Numbers and dates keep their types.
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Data', index=False)

        # Create issues summary sheet
        issues_df = pd.DataFrame(issues)
        if not issues_df.empty:
            issues_df.to_excel(writer, sheet_name='Issues', index=False)

        # Rule coverage: a partial result must be visible in the file too.
        if rule_coverage or types_not_checked:
            rule_coverage = rule_coverage or {"complete": True}
            cov_rows = [
                {"item": "Result", "rule_id": "",
                 "detail": "COMPLETE" if rule_coverage.get("complete") else
                           f"PARTIAL - {rule_coverage.get('rules_not_checked', 0)} of "
                           f"{rule_coverage.get('rules_total', 0)} rules could not be checked"},
                {"item": "Rules total", "rule_id": "", "detail": rule_coverage.get("rules_total", 0)},
                {"item": "Rules evaluated", "rule_id": "", "detail": rule_coverage.get("rules_evaluated", 0)},
                {"item": "Rules not checked", "rule_id": "", "detail": rule_coverage.get("rules_not_checked", 0)},
            ]
            for kind, label in (("unsupported_rules", "Unsupported"), ("failed_rules", "Failed")):
                for r in rule_coverage.get(kind, []):
                    cov_rows.append({"item": label, "rule_id": r.get("rule_id", ""), "detail": r.get("reason", "")})
            for name in (types_not_checked or []):
                cov_rows.append({"item": "Column type not checked", "rule_id": name,
                                 "detail": "type could not be mapped; type not checked"})
            pd.DataFrame(cov_rows).to_excel(writer, sheet_name='Rule coverage', index=False)

        # Get workbook and worksheet
        workbook = writer.book
        worksheet = workbook['Data']

        # Apply highlighting to cells with issues
        from openpyxl.styles import PatternFill, Font

        error_fill = PatternFill(start_color="FFCCCB", end_color="FFCCCB", fill_type="solid")
        warning_fill = PatternFill(start_color="FFE5B4", end_color="FFE5B4", fill_type="solid")

        for issue in issues:
            if 'row' in issue and 'column' in issue:
                try:
                    col_idx = df.columns.get_loc(issue['column']) + 1
                    row_idx = issue['row'] + 2  # +2 for header and 0-index
                    cell = worksheet.cell(row=row_idx, column=col_idx)

                    if issue['severity'] == 'error':
                        cell.fill = error_fill
                    elif issue['severity'] == 'warning':
                        cell.fill = warning_fill

                    # Add comment with issue details
                    from openpyxl.comments import Comment
                    cell.comment = Comment(str(issue['message']), "Data Analyzer")
                except Exception:
                    pass

        # Neutralise formula-like text on EVERY sheet (data, headers, issues)
        force_text_cells(workbook)

        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except Exception:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width

    return output.getvalue()

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_source_format' not in st.session_state:
    # Tracks the original upload format ('csv', 'json', 'tsv', 'xlsx', 'xls')
    # so the analysis HTTP call can re-serialize st.session_state.data with a
    # matching content-type instead of always defaulting to CSV.
    st.session_state.data_source_format = None
if 'dictionary' not in st.session_state:
    st.session_state.dictionary = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'mcp_client' not in st.session_state:
    st.session_state.mcp_client = DataQualityAnalyzer()
if st.session_state.mcp_client.config_error:
    # Deployed without DATA_ANALYZER_API_URL: say so up front, one line, rather
    # than letting "Analyze" fail later. (entrypoint.sh normally stops such a
    # container from starting at all; this covers other launch paths.)
    st.error(f"⚠️ {st.session_state.mcp_client.config_error}")
if 'dict_cache' not in st.session_state:
    st.session_state.dict_cache = {}  # Cache for parsed dictionaries
if 'last_dict_file' not in st.session_state:
    st.session_state.last_dict_file = None  # Track last uploaded dictionary
if 'cache_session_id' not in st.session_state:
    # Random per-browser-session id: namespaces this session's dictionary cache
    # so other users' sessions can neither read, count nor delete its entries.
    st.session_state.cache_session_id = secrets.token_hex(16)
if 'dict_cache_store' not in st.session_state:
    try:
        st.session_state.dict_cache_store = DictionaryCache(
            default_cache_base(), st.session_state.cache_session_id)
        st.session_state.dict_cache_store.prune()  # drop expired entries of any session
    except OSError:
        logger.warning("dictionary_cache_unavailable", exc_info=True)
        st.session_state.dict_cache_store = None  # fall back to the in-memory cache only


if nav_choice == "Analyze":
    # Subtitle only - title is in the navbar
    st.markdown("Upload your data, optionally add validation rules, and analyze")

    # Create three columns for the main components
    col1, col2, col3 = st.columns([2, 2, 1.5])

    with col1:
        with st.container(key="card-upload-data"):
            st.markdown('<div class="app-section-title">Upload Data</div>', unsafe_allow_html=True)


            # File uploader first
            uploaded_file = st.file_uploader(
                " ",  # Empty label to avoid duplication
                type=['csv', 'json', 'txt', 'xlsx', 'xls'],
                key="data_uploader",
                label_visibility="collapsed"
            )

            if uploaded_file and uploaded_file.size > max_upload_bytes():
                st.error(f"This file is {uploaded_file.size / (1024 * 1024):.1f} MB; the upload limit is "
                         f"{max_upload_bytes() // (1024 * 1024)} MB. Split the file or contact the app owner.")
                uploaded_file = None
            if uploaded_file:
                with st.spinner("Processing the uploaded data file..."):
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            st.session_state.data = pd.read_csv(uploaded_file)
                            st.session_state.data_source_format = 'csv'
                        elif uploaded_file.name.endswith('.json'):
                            st.session_state.data = pd.read_json(uploaded_file)
                            st.session_state.data_source_format = 'json'
                        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                            # Shared loader (mcp_server.DataLoader) so MCP server, CLI,
                            # and this UI all get identical Excel handling/error semantics.
                            st.session_state.data = DataLoader.load_excel(uploaded_file.read())
                            # Remember the Excel sub-format so the analysis HTTP
                            # call (DataQualityAnalyzer._call_analyze_api) can
                            # re-upload real Excel bytes with the correct
                            # content-type instead of defaulting to CSV.
                            st.session_state.data_source_format = 'xlsx' if uploaded_file.name.endswith('.xlsx') else 'xls'
                        else:
                            st.session_state.data = pd.read_csv(uploaded_file, sep='\t')
                            st.session_state.data_source_format = 'tsv'
                        st.success(f"✅ Loaded {len(st.session_state.data)} rows × {len(st.session_state.data.columns)} columns")
                    except Exception as e:
                        logger.error("data_load_failed error_class=%s %s", type(e).__name__,
                                     file_fingerprint(uploaded_file.name, uploaded_file.size))
                        st.error(f"Error loading file: {md_escape(e)}")

            # Demo data selector below file uploader
            demo_option = st.selectbox(
                "Or load demo data:",
                ["None", "CSV - Western", "CSV - Asian", "CSV - Clinical", "JSON - Mixed", "REDCap - Clinical (synthetic)"],
                key="demo_selector",
                help="Clinical data includes matching dictionary in demo_data/clinical_dict.json"
            )

            if demo_option != "None":
                dataset_map = {
                    "CSV - Western": "western",
                    "CSV - Asian": "asian",
                    "CSV - Clinical": "clinical",
                    "JSON - Mixed": "mixed",
                    "REDCap - Clinical (synthetic)": "redcap_clinical"
                }
                if demo_option in dataset_map:
                    st.session_state.data = load_demo_data(dataset_map[demo_option])
                    # Demo datasets are always synthesized in-memory (not loaded
                    # from an Excel file), so fall back to CSV serialization for
                    # the analysis HTTP call.
                    st.session_state.data_source_format = 'csv'
                    st.success(f"✅ Loaded {demo_option} demo data")
                    if demo_option == "CSV - Clinical":
                        st.info("📖 Matching dictionary available: Upload 'demo_data/clinical_dict.json' for validation rules")
                    if demo_option == "REDCap - Clinical (synthetic)":
                        st.info("📖 Matching dictionary available: select 'REDCap - Clinical (synthetic, with branching logic)' under 'Or load demo dictionary' below")

    with col2:
        with st.container(key="card-dictionary"):
            st.markdown('<div class="app-section-title">Dictionary</div>', unsafe_allow_html=True)


            # Dictionary file uploader first - aligned with data uploader
            dict_file = st.file_uploader(
                " ",  # Empty label to avoid duplication
                type=['json', 'pdf', 'csv', 'txt', 'xlsx', 'xls'],
                key="dict_uploader",
                label_visibility="collapsed",
                help="Optional - defines validation rules for data quality checks (JSON, PDF, CSV, Excel, or TXT)"
            )

            # LLM model selector - visible whenever AI-assisted dictionary
            # parsing is available at all (not just after a file is uploaded),
            # so it's already set by the time the user uploads a file that
            # needs it. Defaults to the first configured deployment.
            if LLM_AVAILABLE:
                # Only models on the server-side allow-list (AZURE_OPENAI_DEPLOYMENTS)
                # are offered, and the same list is enforced again where the LLM call
                # is made. A value left in the session from a model that is no longer
                # listed would make the selectbox raise, so fall back to the first entry.
                _deployment_options = get_available_deployments()
                st.session_state["llm_deployment_select"] = resolve_deployment_choice(
                    st.session_state.get("llm_deployment_select"), _deployment_options
                )
                st.session_state.llm_deployment = st.selectbox(
                    "LLM model",
                    _deployment_options,
                    key="llm_deployment_select",
                    help="Azure OpenAI deployment used for AI-assisted dictionary parsing. "
                         "Only models enabled on this server are listed; the first is the default.",
                )
                st.caption(f"Model in use for AI parsing: **{st.session_state.llm_deployment}**")

            if dict_file and dict_file.size > max_upload_bytes():
                st.error(f"This dictionary is {dict_file.size / (1024 * 1024):.1f} MB; the upload limit is "
                         f"{max_upload_bytes() // (1024 * 1024)} MB.")
                dict_file = None

            # Add LLM parsing option if available with auto-detection
            if LLM_AVAILABLE and dict_file:
                st.caption(f"📎 Uploaded file: **{md_escape(dict_file.name)}**")

                llm_mode = st.selectbox(
                    "Dictionary parsing method:",
                    ["Auto-detect (recommended)", "Always use AI parsing", "Never use AI (manual only)"],
                    index=0,  # Default to auto-detect
                    help="Auto: PDF→AI, structured CSV→manual | Always: Force AI for all | Never: Manual parsing only"
                )

                # Determine if we should use LLM based on mode and file type
                if "Always" in llm_mode:
                    use_llm = True
                elif "Never" in llm_mode:
                    use_llm = False
                else:  # Auto-detect
                    # Check file type and structure
                    logger.info("dictionary_auto_detect %s", file_fingerprint(dict_file.name, dict_file.size))

                    if dict_file.name.endswith('.pdf'):
                        use_llm = True
                        st.info("🤖 Auto-detected: PDF requires AI parsing")
                    elif dict_file.name.endswith('.csv'):
                        # Peek at CSV to check if it's structured
                        dict_file.seek(0)
                        sample = dict_file.read(1024).decode('utf-8', errors='ignore')
                        dict_file.seek(0)
                        # Check for standard column names
                        if any(col in sample for col in ['Column', 'Type', 'Min', 'Max', 'Allowed_Values', 'Field Name']):
                            use_llm = False
                            st.info("📊 Auto-detected: Structured CSV, using manual parsing")
                        else:
                            use_llm = True
                            st.info("🤖 Auto-detected: Unstructured CSV, using AI parsing")
                    elif dict_file.name.endswith(('.xlsx', '.xls')):
                        # Peek at Excel column headers to check if it's structured
                        # (same convention as the CSV structured-column check above)
                        dict_file.seek(0)
                        try:
                            peek_df = DataLoader.load_excel(dict_file.read())
                            columns_str = ' '.join(str(c) for c in peek_df.columns)
                        except Exception:
                            columns_str = ""
                        dict_file.seek(0)
                        if any(col in columns_str for col in ['Column', 'Type', 'Min', 'Max', 'Allowed_Values', 'Field Name']):
                            use_llm = False
                            st.info("📊 Auto-detected: Structured Excel, using manual parsing")
                        else:
                            use_llm = True
                            st.info("🤖 Auto-detected: Unstructured Excel, using AI parsing")
                    else:
                        use_llm = True
                        st.info(f"🤖 Auto-detected: {md_escape(dict_file.name.split('.')[-1].upper())} file, using AI parsing")
            else:
                use_llm = False

            if dict_file:
                try:
                    # Calculate file hash for caching (works for all file types)
                    dict_file.seek(0)
                    raw_content = dict_file.read()
                    file_hash = hashlib.sha256(raw_content).hexdigest()
                    dict_file.seek(0)  # Reset for reading

                    # Create cache key with LLM flag. Disk entries are private to
                    # this browser session (see DictionaryCache); None = memory only.
                    cache_key = f"{file_hash}_llm" if use_llm else file_hash
                    _store = st.session_state.get("dict_cache_store")
                    _disk_hit = _store.get(cache_key) if _store else None

                    # Check if already cached
                    if cache_key in st.session_state.dict_cache:
                        # Use in-memory cache
                        st.session_state.dictionary = st.session_state.dict_cache[cache_key]

                        logger.info("dictionary_cache_hit tier=memory %s", file_fingerprint(dict_file.name, len(raw_content)))

                        st.success("⚡ Using cached dictionary (instant load)")
                        st.warning("🔄 **CACHE HIT**: Using previously parsed dictionary. Clear cache below if data dictionary changed.")

                        if use_llm:
                            fields_count = len(st.session_state.dictionary.get('fields', []))
                            st.info(f"📊 Contains {fields_count} AI-extracted field definitions")
                        else:
                            st.info(f"📊 Contains {len(st.session_state.dictionary.get('rules', {}))} validation rules")
                    elif _disk_hit is not None:
                        # Per-session JSON cache entry (JSON only, never shared)
                        st.session_state.dictionary = _disk_hit
                        st.session_state.dict_cache[cache_key] = st.session_state.dictionary
                        logger.info("dictionary_cache_hit tier=disk %s", file_fingerprint(dict_file.name, len(raw_content)))

                        st.success(f"⚡ Loaded dictionary from disk cache (no API calls)")
                        st.warning("🔄 **CACHE HIT**: Using previously parsed dictionary. Clear cache below if data dictionary changed.")

                        if use_llm:
                            fields_count = len(st.session_state.dictionary.get('fields', []))
                            st.info(f"📊 Contains {fields_count} AI-extracted field definitions")
                        else:
                            st.info(f"📊 Contains {len(st.session_state.dictionary.get('rules', {}))} validation rules")
                    # Use LLM parsing if enabled and not cached
                    elif use_llm and LLM_AVAILABLE:
                        # CLEAR LLM MARKER
                        _model_label = st.session_state.get("llm_deployment") or llm_model_label()
                        st.warning(f"🤖 **LLM ACTIVE**: Sending data to Azure OpenAI ({_model_label}) for intelligent dictionary parsing...")
                        logger.info("llm_dictionary_parse_started size_bytes=%d", len(raw_content))

                        # More informative spinner with warning about processing time
                        with st.spinner("🤖 Using AI to extract field definitions... This may take 30-60 seconds for large PDFs."):
                            # Read file content
                            file_content = ""
                            if dict_file.name.endswith('.pdf'):
                                # Bounded pages/characters/time/memory, in a child process;
                                # raises DictionaryTooLargeError
                                file_content = extract_pdf_text(dict_file.getvalue())
                            elif dict_file.name.endswith('.csv'):
                                file_content = dict_file.read().decode('utf-8')
                            elif dict_file.name.endswith('.txt'):
                                file_content = dict_file.read().decode('utf-8')
                            elif dict_file.name.endswith('.json'):
                                # For JSON, convert to readable text
                                json_data = json.load(dict_file)
                                file_content = json.dumps(json_data, indent=2)
                            elif dict_file.name.endswith(('.xlsx', '.xls')):
                                # For Excel, convert the sheet to readable text (same
                                # approach as the JSON branch above - binary content
                                # can't be decoded as utf-8 like the CSV/TXT branches)
                                excel_df = DataLoader.load_excel(dict_file.read())
                                file_content = excel_df.to_csv(index=False)
                            else:
                                file_content = dict_file.read().decode('utf-8')

                            check_dictionary_text_size(file_content)

                            # Initialize LLM parser
                            llm_parser = LLMDictionaryParser()

                            estimated_tokens = llm_parser.count_tokens(file_content)
                            start_time = time.time()
                            logger.info("llm_dictionary_parse_input model=%s tokens=%s chars=%d",
                                        llm_parser.deployment, estimated_tokens, len(file_content))

                            # Parse with LLM
                            # Don't truncate - the LLM parser handles chunking internally
                            # Process more fields for comprehensive extraction
                            # One telemetry action per upload. The Easy Auth user
                            # header is trusted only with the same opt-in dcri_llm's
                            # middleware uses (TRUST_EASY_AUTH_HEADER=1, set only once
                            # Easy Auth strips client-sent copies); else anonymous.
                            # Telemetry only, not authorization.
                            _principal = None
                            if (os.environ.get("TRUST_EASY_AUTH_HEADER") or "").strip().lower() in ("1", "true", "yes"):
                                try:
                                    _principal = st.context.headers.get("X-Ms-Client-Principal-Name")
                                except Exception:
                                    _principal = None
                            set_user(_principal)
                            with llm_action("ui_parse_dictionary", new_action=True):
                                parsed_result = llm_parser.parse_dictionary(
                                    file_content,
                                    max_fields=500,
                                    deployment=st.session_state.get("llm_deployment"),
                                )

                            # Calculate elapsed time
                            elapsed_time = time.time() - start_time

                            logger.info("llm_dictionary_parse_done seconds=%.1f fields=%d chunks=%s",
                                        elapsed_time, len(parsed_result.get('fields', [])),
                                        parsed_result.get('metadata', {}).get('chunks_processed', 0))

                            # Store the parsed dictionary
                            st.session_state.dictionary = {
                                "source": "LLM Parser",
                                "filename": dict_file.name,
                                "rules": parsed_result.get("schema", {}),
                                "fields": parsed_result.get("fields", []),
                                "metadata": parsed_result.get("metadata", {})
                            }

                            # Cache the result both in memory and to disk
                            st.session_state.dict_cache[cache_key] = st.session_state.dictionary
                            if _store:
                                _store.put(cache_key, st.session_state.dictionary)
                            st.info("💾 Dictionary cached for this session - re-loading it will be instant (no API calls)")

                            # Add processing time to success message
                            processing_time = parsed_result.get('metadata', {}).get('processing_time_seconds', 0)
                            chunks_processed = parsed_result.get('metadata', {}).get('chunks_processed', 0)
                            st.success(f"✅ AI extracted {len(parsed_result.get('fields', []))} field definitions from {chunks_processed} chunks in {processing_time:.1f} seconds")

                            # Show extracted fields
                            if parsed_result.get('fields'):
                                # Expand by default if we got results, especially for large dictionaries
                                expand_fields = len(parsed_result['fields']) <= 20
                                with st.expander(f"📋 Extracted Fields ({len(parsed_result['fields'])})", expanded=expand_fields):
                                    for field in parsed_result['fields'][:10]:
                                        if isinstance(field, dict):
                                            st.markdown(field_markdown(field))
                                    if len(parsed_result['fields']) > 10:
                                        st.info(f"📊 Showing first 10 of {len(parsed_result['fields'])} extracted fields. Use 'View All Fields' below to see more.")

                    elif dict_file.name.endswith('.pdf'):
                        # PDF without LLM. (Cache hits for this key were already
                        # handled by the branches above.)
                        dict_file.seek(0)  # Reset for reading

                        st.info("📄 **MANUAL PARSING**: Using basic text patterns (limited extraction). Enable AI parsing for better results.")
                        logger.info("manual_pdf_parse_started %s", file_fingerprint(dict_file.name, len(raw_content)))

                        # Parse PDF dictionary with container to prevent UI blocking
                        with st.container():
                            progress_bar = st.progress(0, text="Parsing PDF dictionary...")

                            # Read PDF content. Page count and total characters are
                            # bounded (UI_PDF_MAX_PAGES / UI_DICT_MAX_CHARS); a
                            # DictionaryTooLargeError is shown to the user below.
                            extracted_text, num_pages = extract_pdf_text_and_pages(dict_file.getvalue())
                            progress_bar.progress(1.0, text=f"Processed {num_pages} pages")

                            # Linear-time pattern extraction (no ReDoS, see extract_pdf_manual_rules)
                            extracted_rules = extract_pdf_manual_rules(extracted_text)

                            # Clear progress bar
                            progress_bar.empty()

                        # Store extracted dictionary
                        st.session_state.dictionary = {
                            "source": "PDF",
                            "filename": dict_file.name,
                            "rules": extracted_rules,
                            "pages": num_pages,
                            "text_length": len(extracted_text),
                            "hash": file_hash
                        }

                        # Cache the parsed dictionary in memory and (per session) on disk
                        st.session_state.dict_cache[file_hash] = st.session_state.dictionary
                        if _store:
                            _store.put(file_hash, st.session_state.dictionary)

                        st.success(f"✅ Parsed {num_pages} pages from PDF dictionary")
                        st.info("💾 Dictionary cached for this session")

                        if extracted_rules:
                            with st.expander(f"Found {len(extracted_rules)} validation rules", expanded=False):
                                for field, rule in list(extracted_rules.items())[:10]:  # Show first 10
                                    st.text(f"{field}: {rule}")
                                if len(extracted_rules) > 10:
                                    st.text(f"... and {len(extracted_rules) - 10} more")
                    elif dict_file.name.endswith('.json'):
                        st.session_state.dictionary = validate_dictionary_shape(json.load(dict_file))
                        st.success("✅ JSON dictionary loaded")
                    elif dict_file.name.endswith('.csv') and not use_llm:
                        # Parse CSV dictionary (NO LLM) - only if LLM mode not active
                        st.info("📊 **CSV PARSING**: Reading structured CSV data dictionary...")
                        logger.info("csv_dictionary_parse %s", file_fingerprint(dict_file.name, len(raw_content)))

                        import pandas as pd
                        dict_file.seek(0)
                        df = pd.read_csv(dict_file)
                        rules = {}
                        for _, row in df.iterrows():
                            if 'Column' in row or 'column' in row or 'Field' in row or 'field' in row:
                                field_name = row.get('Column') or row.get('column') or row.get('Field') or row.get('field')
                                if field_name:
                                    rule = {}
                                    if 'Type' in row or 'type' in row:
                                        rule['type'] = str(row.get('Type') or row.get('type'))
                                    if 'Min' in row or 'min' in row:
                                        rule['min'] = row.get('Min') or row.get('min')
                                    if 'Max' in row or 'max' in row:
                                        rule['max'] = row.get('Max') or row.get('max')
                                    if 'Required' in row or 'required' in row:
                                        rule['required'] = row.get('Required') or row.get('required')
                                    if 'Allowed_Values' in row or 'allowed_values' in row:
                                        allowed = row.get('Allowed_Values') or row.get('allowed_values')
                                        if allowed and not pd.isna(allowed):
                                            rule['allowed_values'] = [v.strip() for v in str(allowed).split(',')]
                                    rules[field_name] = rule
                        st.session_state.dictionary = {
                            "source": "CSV",
                            "filename": dict_file.name,
                            "rules": rules
                        }
                        st.success(f"✅ Parsed {len(rules)} field definitions from CSV")
                    elif dict_file.name.endswith(('.xlsx', '.xls')) and not use_llm:
                        # Parse Excel dictionary (NO LLM) - structured field definitions,
                        # same column-name conventions as the CSV dictionary branch above.
                        st.info("📊 **EXCEL PARSING**: Reading structured Excel data dictionary...")
                        logger.info("excel_dictionary_parse %s", file_fingerprint(dict_file.name, len(raw_content)))

                        dict_file.seek(0)
                        df = DataLoader.load_excel(dict_file.read())
                        rules = {}
                        for _, row in df.iterrows():
                            if 'Column' in row or 'column' in row or 'Field' in row or 'field' in row:
                                field_name = row.get('Column') or row.get('column') or row.get('Field') or row.get('field')
                                if field_name:
                                    rule = {}
                                    if 'Type' in row or 'type' in row:
                                        rule['type'] = str(row.get('Type') or row.get('type'))
                                    if 'Min' in row or 'min' in row:
                                        rule['min'] = row.get('Min') or row.get('min')
                                    if 'Max' in row or 'max' in row:
                                        rule['max'] = row.get('Max') or row.get('max')
                                    if 'Required' in row or 'required' in row:
                                        rule['required'] = row.get('Required') or row.get('required')
                                    if 'Allowed_Values' in row or 'allowed_values' in row:
                                        allowed = row.get('Allowed_Values') or row.get('allowed_values')
                                        if allowed and not pd.isna(allowed):
                                            rule['allowed_values'] = [v.strip() for v in str(allowed).split(',')]
                                    rules[field_name] = rule
                        st.session_state.dictionary = {
                            "source": "Excel",
                            "filename": dict_file.name,
                            "rules": rules
                        }
                        st.success(f"✅ Parsed {len(rules)} field definitions from Excel")
                    else:
                        st.error(f"⚠️ Unsupported dictionary format: **{md_escape(dict_file.name)}**")
                        logger.warning("dictionary_unsupported_format use_llm=%s llm_available=%s %s",
                                       use_llm, LLM_AVAILABLE, file_fingerprint(dict_file.name, len(raw_content)))
                except DictionaryTooLargeError as e:
                    # Our own limit message (page/character caps): safe and actionable.
                    logger.warning("dictionary_too_large %s", file_fingerprint(dict_file.name, dict_file.size))
                    st.error(f"⚠️ {md_escape(e)}")
                except Exception as e:
                    logger.error("dictionary_load_failed error_class=%s %s", type(e).__name__,
                                 file_fingerprint(dict_file.name, dict_file.size))
                    st.error(f"Error loading dictionary: {md_escape(e)}")

            # View All Fields button - accessible location near dictionary upload
            if st.session_state.dictionary and st.session_state.dictionary.get('fields'):
                fields_list = st.session_state.dictionary['fields']
                if not isinstance(fields_list, list):
                    fields_list = []
                num_fields = len(fields_list)

                if num_fields > 0:
                    st.markdown("---")
                    with st.expander(f"📋 View All {num_fields} Extracted Fields", expanded=False):
                        for field in fields_list:
                            if isinstance(field, dict):
                                st.markdown(field_markdown(field, detailed=True, max_values=8))

            # Demo dictionary selector below file uploader
            demo_dict = st.selectbox(
                "Or load demo dictionary:",
                ["None"] + list(DEMO_DICTIONARIES.keys()),
                key="demo_dict_selector"
            )

            demo_dictionary_built = None
            if demo_dict != "None":
                # Get demo dictionary CSV string and parse it
                demo_csv_string = get_demo_dictionary(demo_dict)

                # Parse CSV string into rules dictionary (same logic as CSV upload)
                import io
                df = pd.read_csv(io.StringIO(demo_csv_string))
                rules = {}
                for _, row in df.iterrows():
                    if 'Column' in row or 'column' in row or 'Field' in row or 'field' in row:
                        field_name = row.get('Column') or row.get('column') or row.get('Field') or row.get('field')
                        if field_name:
                            rule = {}
                            if 'Type' in row or 'type' in row:
                                rule['type'] = str(row.get('Type') or row.get('type'))
                            if 'Min' in row or 'min' in row:
                                rule['min'] = row.get('Min') or row.get('min')
                            if 'Max' in row or 'max' in row:
                                rule['max'] = row.get('Max') or row.get('max')
                            if 'Required' in row or 'required' in row:
                                rule['required'] = row.get('Required') or row.get('required')
                            if 'Allowed_Values' in row or 'allowed_values' in row:
                                allowed = row.get('Allowed_Values') or row.get('allowed_values')
                                if allowed and not pd.isna(allowed):
                                    rule['allowed_values'] = [v.strip() for v in str(allowed).split(',')]
                            rules[field_name] = rule

                demo_dictionary_built = {
                    "source": "Demo Dictionary",
                    "filename": demo_dict,
                    "rules": rules
                }
                st.success(f"✅ Loaded {demo_dict} ({len(rules)} field definitions)")

            # Centralized decision for what st.session_state.dictionary should be
            # this render pass (fixes stale-dictionary bug: resetting the demo
            # selector to "None" used to leave a previously-loaded demo
            # dictionary's rules active forever, since there was no `else` to
            # clear them). See resolve_effective_dictionary() docstring for the
            # full precedence rules (upload > demo selection > None).
            st.session_state.dictionary = resolve_effective_dictionary(
                demo_dict_selection=demo_dict,
                demo_dictionary_built=demo_dictionary_built,
                dict_file_uploaded=dict_file is not None,
                previous_session_dictionary=st.session_state.dictionary,
            )

            # Add cache management
            st.markdown("---")
            st.markdown("#### Cache Management")

            # Only THIS session's entries are counted or deleted (no server paths,
            # no other users' activity).
            _store = st.session_state.get("dict_cache_store")
            num_cached = _store.count() if _store else 0

            if num_cached > 0:
                st.caption(f"📦 {num_cached} dictionaries cached in your session")

                if st.button("🗑️ Clear my cache", help="Delete your session's cached dictionaries to force re-parsing"):
                    try:
                        removed = _store.clear()
                        st.session_state.dict_cache = {}
                        st.session_state.dictionary = None
                        st.success(f"✅ Cleared {removed} cached dictionaries")
                        st.rerun()
                    except OSError as e:
                        logger.error("dictionary_cache_clear_failed error_class=%s", type(e).__name__)
                        st.error("❌ Could not clear the cache.")
            else:
                st.caption("No cached dictionaries")

    with col3:
        # The Export controls are drawn into this slot AFTER the Analyze handler below
        # has run: the handler is what sets analysis_results, so drawing here (earlier
        # in the script) would skip Export in the very run that finishes an analysis.
        export_slot = st.empty()

    # Run Analysis - full width between uploads and results
    st.markdown("<div style='margin: 1.75rem 0 1rem 0; border-top: 1px solid #e2e8f0;'></div>", unsafe_allow_html=True)

    # Create centered column for button
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        # Enable button only when data is loaded
        if st.button(
            "Analyze Data Quality",
            disabled=(st.session_state.data is None),
            use_container_width=True,
            type="primary",
            help="Load data first, then click to analyze" if st.session_state.data is None else "Run comprehensive quality checks on your data",
            key="run_analysis_main"
        ):
            if st.session_state.data is not None:
                # A new analysis starts from a clean slate: if it fails, no earlier
                # result (or its Export) may stay on the page.
                st.session_state.analysis_results = None
                st.session_state.analysis_fingerprint = None
                # Log dictionary usage
                if st.session_state.dictionary:
                    dict_source = st.session_state.dictionary.get('source', 'Unknown')
                    dict_filename = st.session_state.dictionary.get('filename', 'Unknown')
                    num_rules = len(st.session_state.dictionary.get('rules', {}))
                    num_fields = len(st.session_state.dictionary.get('fields', []))

                    logger.info("analysis_started rows=%d columns=%d dictionary_source=%s rules=%d fields=%d",
                                len(st.session_state.data), len(st.session_state.data.columns),
                                dict_source, num_rules, num_fields)

                    st.info(f"📖 Using dictionary: **{md_escape(dict_filename)}** ({md_escape(dict_source)}) - {num_rules} rules, {num_fields} fields")
                else:
                    logger.info("analysis_started_without_dictionary")
                    st.info("⚠️ No dictionary loaded - using auto-detection only")

                _status_box = st.empty()
                st.session_state.mcp_client.on_status = lambda msg: _status_box.info(f"⏳ {msg}")
                with st.spinner("🔍 Analyzing data quality... (the first run after a quiet period can take up to a minute)"):
                    try:
                        # Run analysis
                        results = asyncio.run(
                            st.session_state.mcp_client.analyze_data_quality(
                                st.session_state.data,
                                st.session_state.dictionary,
                                source_format=st.session_state.get('data_source_format')
                            )
                        )
                        st.session_state.analysis_results = results
                        st.session_state.analysis_fingerprint = analysis_inputs_fingerprint(
                            st.session_state.data, st.session_state.dictionary)
                        _status_box.empty()
                        st.success("✅ Analysis complete!")
                    except (ApiConfigError, RuntimeError) as e:
                        # Expected, already-explained failures: one clear line,
                        # no traceback for the user. Logged for diagnosis.
                        _status_box.empty()
                        # Class only: the text may quote upstream/API detail (cell values).
                        logger.error("analysis_failed error_class=%s", type(e).__name__)
                        st.error(f"❌ Analysis failed: {md_escape(e)}")
                    except Exception as e:
                        _status_box.empty()
                        logger.error("analysis_failed_unexpected error_class=%s\n%s",
                                     type(e).__name__, traceback.format_exc())
                        st.error(f"❌ Analysis failed unexpectedly ({type(e).__name__}). "
                                 "The error was logged; please contact the app owner.")

    st.markdown("<div style='margin: 1.75rem 0 1rem 0; border-top: 1px solid #e2e8f0;'></div>", unsafe_allow_html=True)

    # Results belong to the data + dictionary that were analysed. If either changed
    # since, they are stale: clear them (and say so) instead of showing an old result
    # next to new data.
    if st.session_state.analysis_results and st.session_state.get('analysis_fingerprint') != \
            analysis_inputs_fingerprint(st.session_state.data, st.session_state.dictionary):
        st.session_state.analysis_results = None
        st.session_state.analysis_fingerprint = None
        st.info("The data or dictionary changed since the last analysis, so those results were cleared. "
                "Click **Analyze Data Quality** to analyse again.")

    # Export (drawn here, after the handler, into the slot in the top row)
    if st.session_state.analysis_results:
        with export_slot.container():
            render_export_controls(st.session_state.data, st.session_state.analysis_results)

    # Display results if available
    if st.session_state.analysis_results:

        # Summary metrics
        st.subheader("Analysis Summary")
        summary = st.session_state.analysis_results['summary']

        if st.session_state.analysis_results.get('logic_validation_error'):
            st.warning(f"⚠️ {md_escape(st.session_state.analysis_results['logic_validation_error'])}")

        _untyped = summary.get('column_types_not_checked') or []
        if _untyped:
            st.warning(
                f"⚠️ {len(_untyped)} column type(s) could not be checked: "
                + ", ".join(md_escape(c) for c in _untyped[:20])
                + (f" (and {len(_untyped) - 20} more)" if len(_untyped) > 20 else "")
                + ". These columns were still checked for rules, but not for their type."
            )

        _coverage = st.session_state.analysis_results.get('rule_coverage')
        if _coverage and _coverage.get('rules_not_checked'):
            st.warning(
                f"⚠️ **Partial result: {_coverage['rules_not_checked']} of {_coverage['rules_total']} "
                "conditional rules could not be checked.** Their absence from the issues below does "
                "NOT mean the data passed them."
            )
            with st.expander("Rules that could not be checked"):
                for _kind, _label in (("unsupported_rules", "Unsupported"), ("failed_rules", "Failed during evaluation")):
                    for _r in _coverage.get(_kind, []):
                        st.markdown(f"- {_label}: `{md_escape(_r.get('rule_id'))}` ({md_escape(_r.get('reason'))})")

        # Create columns with space for heatmap
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1.2, 1, 1, 1.5])
        with col1:
            st.metric("Total Rows", f"{summary['total_rows']:,}")
        with col2:
            st.metric("Total Columns", summary['total_columns'])
        with col3:
            st.metric("Issues Found", summary['issues_found'],
                     delta=None if summary['issues_found'] == 0 else f"{summary['critical_issues']} critical")
        with col4:
            st.metric("Warnings", summary['warnings'])
        with col5:
            st.metric("Completeness", f"{summary['completeness']}%")
        with col6:
            # Create issue heatmap visualization
            st.markdown("#### Issue Map")
            create_issue_heatmap(st.session_state.data, st.session_state.analysis_results['issues'])

        # Issues details
        if st.session_state.analysis_results['issues']:
            st.subheader("Issues Found")

            # Group issues by type
            issues_by_type = {}
            for issue in st.session_state.analysis_results['issues']:
                issue_type = issue['type']
                if issue_type not in issues_by_type:
                    issues_by_type[issue_type] = []
                issues_by_type[issue_type].append(issue)

            # Display issues by type - collapsed by default for cleaner look
            for issue_type, issues in issues_by_type.items():
                # Collapse by default for Missing Values and Invalid Values
                expand_by_default = issue_type not in ['missing_values', 'invalid_value']
                _type_totals = summary.get('issue_type_totals') or {}
                true_total = max(len(issues), int(_type_totals.get(issue_type, 0)))
                with st.expander(f"{md_escape(str(issue_type).replace('_', ' ').title())} ({true_total} issues)", expanded=expand_by_default):
                    for issue in issues[:10]:  # Show first 10
                        if issue['severity'] == 'error':
                            st.error(f"❌ {md_escape(issue['message'])}")
                        else:
                            st.warning(f"⚠️ {md_escape(issue['message'])}")
                    if true_total > min(len(issues), 10):
                        st.info(f"Showing the first {min(len(issues), 10)} of {true_total} issues of this kind; "
                                f"{true_total - min(len(issues), 10)} more are counted but not listed.")

        # Recommendations
        if st.session_state.analysis_results['recommendations']:
            st.subheader("Recommendations")
            for rec in st.session_state.analysis_results['recommendations']:
                if rec['priority'] == 'critical':
                    st.error(f"🔴 **{md_escape(rec['priority'].upper())}**: {md_escape(rec['message'])}")
                elif rec['priority'] == 'high':
                    st.warning(f"🟡 **{md_escape(rec['priority'].upper())}**: {md_escape(rec['message'])}")
                else:
                    st.info(f"🔵 **{md_escape(rec['priority'].upper())}**: {md_escape(rec['message'])}")

elif nav_choice == "About":
    st.title("About Data Quality Analyzer")

    st.markdown("""
    ### Purpose
    The Data Quality Analyzer helps you find data quality problems in a dataset so you can
    look at them and fix them. It reports what its checks find. It is a screening aid: it does
    not certify that data is correct, and it has not been validated or approved for use with
    real study data.

    ### Features
    - **Multiple Format Support**: CSV, JSON, Excel (XLSX/XLS), and TXT files
    - **Automatic Issue Detection**: Missing values, values that do not fit a column's type, and values outside the rules you supply
    - **Custom Validation Rules**: Define min/max ranges, allowed values and conditional (branching) rules via a data dictionary (JSON, CSV, Excel or PDF)
    - **Visual Reporting**: Metrics, issue groups and a heatmap of where issues are
    - **Export**: Excel with highlighted cells and comments, or a JSON report. Both include which conditional rules could not be checked
    - **Demo Data**: Built-in example datasets and dictionaries to try the tool
    - **Dictionary Caching**: A parsed dictionary is cached for your browser session only and expires after a few hours

    ### What We Check
    1. **Missing Values**: Counts empty cells per column
    2. **Data Type Validation**: Whole-number, decimal, date/time and true/false columns are checked against the type in your dictionary (or the type detected from the data)
    3. **Range Validation**: Numeric values against the min/max you specify
    4. **Allowed Values**: Values against the list you specify
    5. **Conditional Rules**: Rules such as "if A then B must be blank" taken from the dictionary; rules the tool cannot understand are listed as not checked
    6. **Row Count and Completeness**: Minimum rows, and the overall percentage of non-empty cells

    ### What It Does Not Check
    - Placeholder or junk entries such as "n/a", "invalid", "error" or test data are not looked for
    - There is no outlier or "suspicious value" detection
    - Duplicate rows are only counted in the summary statistics; they are not reported as issues
    - A dictionary's "Required" flag is read but not enforced
    - Text columns are not checked beyond allowed-value lists
    - Anything that depends on a dictionary rule is only as good as that rule; an AI-read dictionary should be reviewed by a person

    ### How to Use
    1. **Upload your data** using the file uploader or select demo data
    2. **Optionally add a dictionary** (JSON, CSV, Excel, PDF or text) to define custom validation rules
    3. **Click Analyze** to run the quality checks
    4. **Review the results** including issues, recommendations, and visual heatmap
    5. **Export findings** to Excel (with highlighting) or JSON for further analysis

    ### Technical Details
    Built with Streamlit. The checks run in the data-analyzer REST API (see below). Features include:
    - Interactive Plotly visualizations
    - PDF parsing with pypdf
    - Excel generation with cell highlighting and comments
    - Per-session dictionary cache

    ### Data Flow Architecture
    """)

    # Load and render the Mermaid diagrams with selector
    try:
        # Load both diagrams
        with open('assets/data_flow_simple.mmd', 'r') as f:
            simple_diagram = f.read()
        with open('assets/data_flow_diagram.mmd', 'r') as f:
            detailed_diagram = f.read()

        st.info("Flowcharts showing the data analysis pipeline:")

        # Add diagram selector within the content area
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            diagram_view = st.radio(
                "Select diagram complexity:",
                ["Simple View", "Detailed View"],
                horizontal=True,
                label_visibility="visible"
            )

        # Render the selected diagram
        if diagram_view == "Simple View":
            render_mermaid(simple_diagram, height=400, image_path='assets/data_flow_simple.png')
        else:
            render_mermaid(detailed_diagram, height=700, image_path='assets/data_flow_diagram.png')

    except FileNotFoundError as e:
        st.info("Data flow diagram files were not found.")
    except Exception as e:
        st.error(f"Error showing the diagram ({type(e).__name__}).")

    st.markdown("---")
    st.header("Use it from code (API)")
    st.markdown(
        "For pipelines and scripts, the REST API is the intended interface "
        "instead of this UI. It requires **(1)** the Duke VPN and **(2)** an "
        "API key sent in the `X-API-Key` header. Request the key from the app "
        "owner, then run `export DATA_ANALYZER_API_KEY=...` in your shell. "
        "Every block below then runs exactly as shown (use the copy button at "
        "the top right of each block). Requests are rate-limited. "
        "Key-less access via your Duke (Entra ID) sign-in is planned."
    )

    # The block text lives in src/api_examples.py (also executed verbatim by
    # tests/test_about_api_examples.py). The key's value is never read here:
    # the blocks only reference $DATA_ANALYZER_API_KEY / os.environ[...].
    _example_url, _example_note = display_base_url(os.getenv("DATA_ANALYZER_API_URL", ""))
    if _example_note:
        st.info(_example_note)
    _examples_by_group = {}
    for _ex in build_api_examples(_example_url):
        _examples_by_group.setdefault(_ex["group"], []).append(_ex)

    for _group_id, _heading, _blurb in EXAMPLE_GROUPS:
        st.subheader(_heading)
        st.caption(_blurb)
        _group = _examples_by_group[_group_id]
        _tabs = st.tabs(["curl" if e["language"] == "bash" else "Python" for e in _group])
        for _tab, _ex in zip(_tabs, _group):
            with _tab:
                st.code(_ex["code"], language=_ex["language"])

    st.markdown(
        f"Interactive API docs (Swagger UI, where you can try `/analyze` in the browser): "
        f"[`{_example_url}/api/v1/docs`]({_example_url}/api/v1/docs). "
        f"Health and version also open directly in a browser. "
        f"Incognito/private browsing only avoids sending cookies - it does **not** "
        f"simulate being off the VPN. To test the VPN requirement itself, use a "
        f"device that is actually off-network (e.g. a phone on cellular data, wifi off)."
    )

    st.markdown(f"""
    ---
    *Version {__version__} - Enhanced UI with PDF Dictionary Support*
    """)
