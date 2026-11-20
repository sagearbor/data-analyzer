"""
FastAPI REST API Server for Data Analyzer

Provides programmatic access to data quality analysis, dictionary parsing,
and validation program management.

Features:
- Data analysis with optional dictionary
- Dictionary parsing and program management
- Logic validation using cached programs
- Rate limiting and error handling
- CORS support for web clients

Usage:
    uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
"""

import base64
import binascii
import contextvars
import hashlib
import io
import json
import logging
import os
import re
import secrets
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

import pandas as pd

from fastapi import FastAPI, HTTPException, Request, status, Depends, UploadFile, File, Form
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from limits import parse as parse_rate_limit
from limits.strategies import FixedWindowRateLimiter
from starlette.concurrency import run_in_threadpool
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from dotenv import load_dotenv

# Import existing modules
#
# Response/request models from src.api_models are required for route registration
# (they appear in @app.get/@app.post response_model= and type hints), so they are
# imported in their own try/except block. If they fail to import, the app cannot
# start at all and that failure must be loud and immediate rather than silently
# leaving names like ParseDictionaryResponse undefined (which previously caused a
# NameError at route-decoration time, crashing the whole server on startup).
from src.api_models import (
    ParseDictionaryRequest,
    ParseDictionaryResponse,
    ProgramDetail,
    convert_validation_program_to_detail,
    AnalyzeResponse,
    AnalysisSummary,
    FieldViolation,
    LogicViolation,
    SeverityEnum,
    DataFormatEnum,
    ReturnFormatEnum,
    QualityAnalysisResponse,
    AnalyzeJsonRequest,
)

# Shared DCRI LLM telemetry (vendored; stdlib-only at import time).
from src.dcri_llm import asgi_middleware, llm_action
from src.runtime_env import is_deployed_env
from src.__version__ import __version__

# Optional service modules - each is independently best-effort. A missing/broken
# module here degrades the corresponding feature (endpoints return 503) instead
# of taking down the whole server.
try:
    from src.llm_client import LLMDictionaryParser
except ImportError as e:
    logging.error(f"Import error (LLMDictionaryParser unavailable): {e}")
    LLMDictionaryParser = None

try:
    from src.program_manager import ProgramManager, ProgramSaveError
except ImportError as e:
    logging.error(f"Import error (ProgramManager unavailable): {e}")
    ProgramManager = None

    class ProgramSaveError(RuntimeError):  # placeholder so except clauses stay valid
        pass

try:
    from src.logic_engine import LogicValidator
except ImportError as e:
    logging.error(f"Import error (LogicValidator unavailable): {e}")
    LogicValidator = None

try:
    import mcp_server
except ImportError as e:
    logging.error(f"Import error (mcp_server unavailable): {e}")
    mcp_server = None

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI Application Setup
# ============================================================================

app = FastAPI(
    title="Data Analyzer REST API",
    description="""
    REST API for data quality analysis with logic validation.

    ## Features

    * **Data Analysis**: Upload CSV/JSON/Excel/Parquet files for quality analysis
    * **Dictionary Parsing**: Extract schemas and validation rules from data dictionaries
    * **Program Management**: Create, retrieve, and manage validation programs
    * **Logic Validation**: Execute conditional validation rules on your data

    ## Authentication

    Most endpoints require API key authentication via the `X-API-Key` header.
    Contact your administrator for an API key.

    ## Rate Limiting

    API endpoints are rate-limited to prevent abuse. Default limits:
    - Health endpoint: 60 requests/minute
    - Other endpoints: 10 requests/minute (configurable per endpoint)
    """,
    version=__version__,
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
)

# Route paths referenced from more than one place (handlers, middleware).
ANALYZE_PATH = "/api/v1/analyze"
ANALYZE_JSON_PATH = "/api/v1/analyze/json"
DICTIONARY_PARSE_PATH = "/api/v1/dictionary/parse"

# ============================================================================
# Middleware Configuration
# ============================================================================

# CORS
#
# allow_origins=["*"] combined with allow_credentials=True lets Starlette's
# CORSMiddleware dynamically reflect back whatever Origin header the browser
# sends while still asserting Access-Control-Allow-Credentials: true - i.e. any
# website can make credentialed cross-origin requests to this API from a
# victim's browser. Origins must be explicitly enumerated instead.
#
# Configure via ALLOWED_ORIGINS env var (comma-separated). In non-prod
# environments, common local dev ports are allowed by default for convenience;
# in prod, nothing is allowed unless explicitly configured.
_allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
if _allowed_origins_env:
    ALLOWED_ORIGINS = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
elif os.getenv("APP_ENV", "dev") == "prod":
    ALLOWED_ORIGINS = []
    logger.warning(
        "ALLOWED_ORIGINS not set in production (APP_ENV=prod). "
        "No cross-origin browser requests will be permitted."
    )
else:
    ALLOWED_ORIGINS = ["http://localhost:3002", "http://localhost:8501", "http://127.0.0.1:3002"]

# Per-request telemetry context (action_id + Easy Auth user) for LLM call events.
app.add_middleware(asgi_middleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Maximum request size, per upload route.
#
# Analyze routes: same setting and 100MB default as
# mcp_server.DataLoader.MAX_CONTENT_BYTES (DATA_ANALYZER_MAX_UPLOAD_BYTES).
# Dictionary parse: a data dictionary is small text and every ~4.5k tokens of it
# costs one LLM call, so it gets its own much smaller cap
# (DATA_ANALYZER_MAX_DICTIONARY_UPLOAD_BYTES, default 2MB).
# Read per request rather than once at import so the limits can be changed
# without re-importing this module.
DEFAULT_MAX_UPLOAD_BYTES = 100 * 1024 * 1024
DEFAULT_MAX_DICTIONARY_UPLOAD_BYTES = 2 * 1024 * 1024

# Every route that accepts a request body of user data, mapped to the env var
# that bounds it. A new upload route MUST be added here.
_UPLOAD_LIMIT_ENV = {
    ANALYZE_PATH: "DATA_ANALYZER_MAX_UPLOAD_BYTES",
    ANALYZE_JSON_PATH: "DATA_ANALYZER_MAX_UPLOAD_BYTES",
    DICTIONARY_PARSE_PATH: "DATA_ANALYZER_MAX_DICTIONARY_UPLOAD_BYTES",
}


def _limit_from_env(env_name: str, default: int) -> int:
    try:
        limit = int(os.getenv(env_name, default))
    except ValueError:
        return default
    return limit if limit > 0 else default


def _max_request_bytes() -> int:
    """Return the configured maximum analyze-request size in bytes."""
    return _limit_from_env("DATA_ANALYZER_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES)


def _max_dictionary_request_bytes() -> int:
    """Return the configured maximum dictionary-upload size in bytes."""
    return _limit_from_env(
        "DATA_ANALYZER_MAX_DICTIONARY_UPLOAD_BYTES", DEFAULT_MAX_DICTIONARY_UPLOAD_BYTES
    )


def _request_limit_for_path(path: str) -> int:
    if path == DICTIONARY_PARSE_PATH:
        return _max_dictionary_request_bytes()
    return _max_request_bytes()


def _too_large_detail(limit: int, env_name: str = "DATA_ANALYZER_MAX_UPLOAD_BYTES") -> str:
    return (
        f"Request too large: the limit is {limit} bytes "
        f"({env_name})."
    )


# Server-generated correlation id for the current request (never taken from the
# client, so it cannot be used to forge log lines or spoof correlation).
_request_id_ctx: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "data_analyzer_request_id", default=None
)


def _current_request_id() -> str:
    """Return the current request's correlation id (generating one if needed)."""
    rid = _request_id_ctx.get()
    if not rid:
        rid = uuid.uuid4().hex[:16]
        _request_id_ctx.set(rid)
    return rid


def _log_safe(value: Any, max_len: int = 64) -> str:
    """Make a caller-supplied string safe to put in a log line (no control
    characters / newlines, bounded length)."""
    text = re.sub(r"[^\x20-\x7e]", "?", str(value))
    return text[:max_len] + ("..." if len(text) > max_len else "")


def _content_fingerprint(content: bytes) -> str:
    """Short hash + size for logging an upload without its name or values."""
    return f"size={len(content)} sha256={hashlib.sha256(content).hexdigest()[:12]}"


class RequestIdMiddleware:
    """Assign a server-generated correlation id to every HTTP request, expose it
    to handlers/log lines through a context variable and return it in the
    X-Request-ID response header. Outermost middleware."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        rid = uuid.uuid4().hex[:16]
        token = _request_id_ctx.set(rid)

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers") or [])
                headers.append((b"x-request-id", rid.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            _request_id_ctx.reset(token)


class MaxRequestSizeMiddleware:
    """Enforce the request size limit on every upload route while the body
    is being received.

    Pure ASGI middleware (not @app.middleware("http"), which only sees the
    request after the body has been handed over): it wraps `receive` and
    counts the bytes of every `http.request` message, so an oversize body is
    refused as soon as the running total passes the limit - whether the
    Content-Length header is truthful, wrong, or absent (chunked upload).
    A truthful oversize Content-Length is rejected up front, before any of
    the body is read.

    Covers the routes listed in _UPLOAD_LIMIT_ENV (both analyze routes and the
    dictionary parse route), each with its own limit.

    Registered ahead of add_security_headers so that middleware still wraps
    the 413 response.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] not in _UPLOAD_LIMIT_ENV
        ):
            await self.app(scope, receive, send)
            return

        limit = _request_limit_for_path(scope["path"])
        env_name = _UPLOAD_LIMIT_ENV[scope["path"]]
        headers = dict(scope.get("headers") or [])
        request_id = _current_request_id()

        async def send_too_large(detail: str):
            response = JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content=ErrorResponse(
                    error=_too_large_detail(limit, env_name),
                    detail=detail,
                    timestamp=datetime.now().isoformat(),
                    request_id=request_id,
                ).dict(),
            )
            await response(scope, receive, send)

        # Fast path: the client says up front that the body is too big.
        try:
            declared = int(headers.get(b"content-length", b"").decode("latin-1"))
        except ValueError:
            declared = None
        if declared is not None and declared > limit:
            await send_too_large(f"Content-Length was {declared} bytes.")
            return

        received = 0
        rejected = False
        response_started = False

        async def counting_receive():
            nonlocal received, rejected
            if rejected:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    # Answer now and stop feeding the body to the app: it
                    # sees a disconnect, and anything it then tries to send
                    # is dropped by guarded_send below.
                    rejected = True
                    if not response_started:
                        await send_too_large(
                            f"Stopped reading after {received} bytes."
                        )
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message):
            nonlocal response_started
            if rejected:
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        await self.app(scope, counting_receive, guarded_send)


app.add_middleware(MaxRequestSizeMiddleware)

# Baseline security response headers (defense-in-depth for browser-based clients
# and for any reverse proxy that forwards these responses as-is).
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers.setdefault("Cache-Control", "no-store")
    return response

# Added last = outermost: every response (including 413/401/429) carries the id.
app.add_middleware(RequestIdMiddleware)

# Rate limiting
#
# get_remote_address (slowapi) keys on request.client.host and does NOT parse
# X-Forwarded-For, so it cannot be spoofed via that header. Note however that
# behind Azure Container Apps ingress / any reverse proxy, request.client.host
# will be the proxy's address for every request unless the proxy is configured
# to preserve the real client IP - in that case all clients share one rate
# limit bucket. Verify the ingress forwards a trustworthy client IP (or accept
# that rate limiting is effectively global) before relying on it for abuse
# prevention.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ============================================================================
# Service Initialization
# ============================================================================

# Initialize services (will be None if imports failed)
llm_client = None
program_manager = None
logic_validator = None

try:
    if LLMDictionaryParser:
        llm_client = LLMDictionaryParser()
        logger.info("LLM Dictionary Parser initialized successfully")

    if ProgramManager and llm_client:
        program_manager = ProgramManager(llm_client)
        logger.info("Program Manager initialized successfully")

    if LogicValidator:
        logic_validator = LogicValidator()
        logger.info("Logic Validator initialized successfully")
except Exception as e:
    logger.error(f"Error initializing services: {e}")
    logger.warning("API will start but some features may be unavailable")

# ============================================================================
# Authentication Configuration
# ============================================================================

# API Key configuration
API_KEY = os.getenv("DATA_ANALYZER_API_KEY")
ADMIN_PASSWORD = os.getenv("DATA_ANALYZER_ADMIN_PASSWORD")

# Fail closed in EVERY non-local environment: missing credentials must abort
# startup, not silently disable authentication (see verify_api_key dev-mode
# fallback). "Non-local" = APP_ENV not in {dev, development, local, test} OR
# running on Azure Container Apps (CONTAINER_APP_NAME is injected by ACA), so
# an ACA manifest that forgets APP_ENV still fails closed.
if is_deployed_env() and not API_KEY:
    raise RuntimeError(
        f"Non-local environment (APP_ENV={os.getenv('APP_ENV', 'dev')!r}, "
        f"CONTAINER_APP_NAME={'set' if os.getenv('CONTAINER_APP_NAME') else 'unset'}) "
        "but DATA_ANALYZER_API_KEY is not set. "
        "Refusing to start with authentication disabled. "
        "Set DATA_ANALYZER_API_KEY (and DATA_ANALYZER_ADMIN_PASSWORD for "
        "admin endpoints) in the container environment."
    )

# Security scheme for API key
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
admin_password_header = APIKeyHeader(name="X-Admin-Password", auto_error=False)

# ---------------------------------------------------------------------------
# Failed-credential throttling
#
# FastAPI resolves Depends(verify_api_key) BEFORE the slowapi decorator on the
# route runs, so the per-route limiter never counts wrong-key requests. Failed
# attempts are therefore counted here, per client, in a fixed window; once a
# client has used up its failures it gets 429 (with Retry-After) for every
# request until the window rolls over - even with the right key, otherwise the
# lockout would not stop guessing.
#
# Client identifier: request.client.host (the TCP peer address). It is NOT read
# from X-Forwarded-For (a client can forge that header). Limitation: behind a
# shared ingress / the Streamlit UI every user shares one peer address, so a
# burst of bad keys from anyone locks that shared address out for the window.
# Tune DATA_ANALYZER_AUTH_FAIL_LIMIT / DATA_ANALYZER_AUTH_FAIL_WINDOW_SECONDS,
# and see the in-code note on the rate limiter about proxy headers.
# Counters live in the slowapi limiter's storage (in memory, per replica).
# ---------------------------------------------------------------------------
DEFAULT_AUTH_FAIL_LIMIT = 10
DEFAULT_AUTH_FAIL_WINDOW_SECONDS = 300

_auth_fail_strategy = FixedWindowRateLimiter(limiter._storage)


def _auth_fail_item():
    limit = _limit_from_env("DATA_ANALYZER_AUTH_FAIL_LIMIT", DEFAULT_AUTH_FAIL_LIMIT)
    window = _limit_from_env("DATA_ANALYZER_AUTH_FAIL_WINDOW_SECONDS", DEFAULT_AUTH_FAIL_WINDOW_SECONDS)
    return parse_rate_limit(f"{limit} per {window} second"), window


def _auth_client_id(request: Optional[Request]) -> str:
    if request is not None and request.client is not None and request.client.host:
        return request.client.host
    return "unknown"


def _reject_if_auth_locked_out(kind: str, request: Optional[Request]) -> None:
    item, window = _auth_fail_item()
    if not _auth_fail_strategy.test(item, "auth-fail", kind, _auth_client_id(request)):
        logger.warning("auth_lockout kind=%s client=%s", kind, _log_safe(_auth_client_id(request)))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed authentication attempts. Try again later.",
            headers={"Retry-After": str(window)},
        )


def _record_auth_failure(kind: str, request: Optional[Request]) -> None:
    item, _window = _auth_fail_item()
    _auth_fail_strategy.hit(item, "auth-fail", kind, _auth_client_id(request))
    logger.warning("auth_failure kind=%s client=%s", kind, _log_safe(_auth_client_id(request)))


def _secrets_equal(supplied: str, expected: str) -> bool:
    """Constant-time comparison that cannot raise on non-ASCII input."""
    return secrets.compare_digest(
        str(supplied).encode("utf-8", "surrogatepass"), expected.encode("utf-8", "surrogatepass")
    )


# Authentication dependency functions
async def verify_api_key(
    api_key: Optional[str] = Depends(api_key_header),
    request: Request = None,
) -> str:
    """
    Verify API key from X-API-Key header

    Args:
        api_key: API key from request header
        request: The request (used to throttle repeated wrong keys per client)

    Returns:
        The validated API key

    Raises:
        HTTPException: 401 if API key is missing, 403 if invalid, 429 after
            too many failed attempts from one client
    """
    # Check if API key authentication is enabled
    if not API_KEY:
        if is_deployed_env():
            # Defense in depth: the start-up guard already refuses to boot like
            # this, but never fall open in a deployed environment.
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication is not configured on the server.",
            )
        logger.warning("API_KEY not configured - authentication disabled")
        return "unauthenticated"

    _reject_if_auth_locked_out("api_key", request)

    # Check if API key is provided
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide valid credentials.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Validate API key (constant-time comparison to avoid timing side-channels)
    if not _secrets_equal(api_key, API_KEY):
        # Count and log the failed attempt (never the key itself)
        _record_auth_failure("api_key", request)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid credentials provided.",
        )

    return api_key

async def verify_admin_password(
    admin_password: Optional[str] = Depends(admin_password_header),
    request: Request = None,
) -> str:
    """
    Verify admin password from X-Admin-Password header

    Fails CLOSED: if DATA_ANALYZER_ADMIN_PASSWORD is not configured (in any
    environment) admin operations are refused with 503 - there is no "open"
    mode and no default password.

    Args:
        admin_password: Admin password from request header
        request: The request (used to throttle repeated wrong passwords)

    Returns:
        The validated admin password

    Raises:
        HTTPException: 503 if not configured, 401 if missing, 403 if invalid,
            429 after too many failed attempts from one client
    """
    if not ADMIN_PASSWORD:
        logger.error("DATA_ANALYZER_ADMIN_PASSWORD not configured - refusing admin request")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin operations are not configured on the server.",
        )

    _reject_if_auth_locked_out("admin", request)

    # Check if admin password is provided
    if not admin_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin authentication required. Provide valid credentials.",
            headers={"WWW-Authenticate": "AdminPassword"},
        )

    # Validate admin password (constant-time comparison to avoid timing side-channels)
    if not _secrets_equal(admin_password, ADMIN_PASSWORD):
        _record_auth_failure("admin", request)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin credentials provided.",
        )

    return admin_password

# ============================================================================
# Pydantic Models (Response schemas)
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: str
    services: Dict[str, bool]

class VersionResponse(BaseModel):
    """Version response"""
    name: str
    version: str

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: Optional[str] = None
    timestamp: str
    request_id: Optional[str] = None

# ============================================================================
# Quality Analysis Report Shaping
# ============================================================================
#
# These helpers reshape the raw dict returned by mcp_server.QualityPipeline
# (checks/summary_stats/issues) into the dashboard-ready shape that
# web_app.py's DataQualityAnalyzer.analyze_data_quality() has always produced
# for the Streamlit UI (summary/issues/recommendations/quality_checks/
# summary_stats). The validation logic itself lives entirely in
# QualityPipeline/QualityChecker (mcp_server.py) - this only reshapes their
# output, so the engine stays the single source of truth. web_app.py now
# calls this endpoint over HTTP instead of importing QualityPipeline
# directly (see DataQualityAnalyzer in web_app.py), so this is the only
# place that shaping happens.

def _json_safe(value: Any) -> Any:
    """Coerce a value that may be a numpy/pandas scalar into a JSON/Pydantic-safe
    native Python value (int, float, bool, str, or None)."""
    if value is None:
        return None
    try:
        import numpy as np  # noqa: PLC0415

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            f = float(value)
            return None if f != f else f  # NaN != NaN
        if isinstance(value, np.bool_):
            return bool(value)
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        # pd.isna() raises on some array-like values; treat those as "not NA"
        pass
    return value


def _json_safe_deep(obj: Any) -> Any:
    """Recursively apply _json_safe() through nested dicts/lists/tuples.

    QualityChecker.get_summary_stats() and run_all_checks() return numpy
    scalar types in places (e.g. Series.to_dict(), .sum()) that pydantic/
    FastAPI's response-model serialization can't encode on their own. This
    walks the full 'quality_checks'/'summary_stats' trees (and anything else
    passed through unchanged from the engine) so the whole report is
    JSON-safe before it hits the response model.
    """
    if isinstance(obj, dict):
        return {k: _json_safe_deep(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe_deep(v) for v in obj]
    return _json_safe(obj)


def _generate_quality_recommendations(issues: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Generate human-readable recommendations from a list of shaped issues.

    Mirrors web_app.py's DataQualityAnalyzer._generate_recommendations exactly
    so the API and (pre-rewire) in-process UI path produce identical output.
    """
    recommendations = []
    issue_types = set(i.get('type', i.get('issue', 'unknown')) for i in issues)

    if 'missing_values' in issue_types:
        recommendations.append({
            "type": "data_cleaning",
            "priority": "high",
            "message": "Consider implementing data imputation strategies for columns with missing values"
        })

    if any(t in issue_types for t in ['type_mismatch', 'datetime_validation_failed', 'invalid_value']):
        recommendations.append({
            "type": "data_validation",
            "priority": "critical",
            "message": "Data type issues detected. Review data source and implement validation at ingestion"
        })

    if 'range_violation' in issue_types or any('violation' in str(t) for t in issue_types):
        recommendations.append({
            "type": "business_rules",
            "priority": "high",
            "message": "Values outside expected ranges detected. Review business rules and data constraints"
        })

    if any('allowed' in str(t) for t in issue_types):
        recommendations.append({
            "type": "categorical_validation",
            "priority": "high",
            "message": "Invalid categorical values found. Verify allowed values match business requirements"
        })

    return recommendations


DEFAULT_MAX_ISSUES_RETURNED = 1000


def _max_issues_returned() -> int:
    """Cap on per-row issues in a response (DATA_ANALYZER_MAX_ISSUES_RETURNED)."""
    return _limit_from_env("DATA_ANALYZER_MAX_ISSUES_RETURNED", DEFAULT_MAX_ISSUES_RETURNED)


def _build_quality_report(df: "pd.DataFrame", results: Dict[str, Any]) -> Dict[str, Any]:
    """Shape a raw QualityPipeline.run_all_checks() result into the
    dashboard-friendly report the web UI expects.

    Parameters:
        df: The loaded DataFrame that was analyzed (needed to look up
            violating cell values and missing-value counts).
        results: The dict returned by QualityPipeline.run_all_checks()
            (has 'issues', 'checks', 'summary_stats' keys).

    Returns:
        {"summary", "issues", "recommendations", "quality_checks", "summary_stats"}
    """
    issues: List[Dict[str, Any]] = []
    max_issues = _max_issues_returned()
    issues_total = 0  # every issue found, including any not returned
    row_errors = 0    # error-severity per-row issues (all of them, returned or not)

    # Transform QualityPipeline issues (which have 'column', 'rule', 'violating_rows')
    # into the UI-shaped format ('type', 'severity', 'message', 'row', 'value').
    for qp_issue in results.get('issues', []):
        column = qp_issue.get('column')
        rule = qp_issue.get('rule', '')
        violating_rows = qp_issue.get('violating_rows', [])

        if 'min >=' in rule or 'max <=' in rule:
            issue_type = "range_violation"
            severity = "error"
        elif 'allowed_values' in rule:
            issue_type = "invalid_categorical_value"
            severity = "error"
        elif qp_issue.get('issue') == 'type_mismatch':
            issue_type = "type_mismatch"
            severity = "error"
        elif qp_issue.get('issue') == 'datetime_validation_failed':
            issue_type = "invalid_date"
            severity = "error"
        else:
            issue_type = qp_issue.get('issue', 'validation_error')
            severity = "error"

        for row_idx in violating_rows:
            issues_total += 1
            row_errors += 1
            if len(issues) >= max_issues:
                continue  # counted, but not returned (see issues_truncated)
            value = df[column].iloc[row_idx] if column in df.columns and row_idx < len(df) else None
            value = _json_safe(value)

            issues.append({
                "type": issue_type,
                "severity": severity,
                "column": column,
                "row": int(row_idx),
                "value": value,
                "message": f"Value {value} in column '{column}' violates rule: {rule}"
            })

    # Add missing-value issues (not produced by QualityChecker itself).
    for col in df.columns:
        missing = int(df[col].isnull().sum())
        if missing > 0:
            issues_total += 1
            issues.append({
                "type": "missing_values",
                "severity": "warning",
                "column": col,
                "count": missing,
                "percentage": round(missing / len(df) * 100, 2),
                "message": f"Column '{col}' has {missing} missing values ({round(missing / len(df) * 100, 2)}%)"
            })

    total_cells = len(df) * len(df.columns)
    completeness = (
        round((1 - df.isnull().sum().sum() / total_cells) * 100, 2) if total_cells else 100.0
    )

    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "issues_found": issues_total,
        "critical_issues": row_errors,
        "warnings": sum(1 for i in issues if i.get('severity') == 'warning'),
        "data_types": results.get('summary_stats', {}).get('dtypes', {col: str(df[col].dtype) for col in df.columns}),
        "completeness": completeness
    }

    report = {
        "summary": summary,
        "issues": issues,
        "issues_total": issues_total,
        "issues_returned": len(issues),
        "issues_truncated": len(issues) < issues_total,
        "recommendations": _generate_quality_recommendations(issues),
        "quality_checks": results.get('checks', {}),
        "summary_stats": results.get('summary_stats', {})
    }
    # QualityChecker.get_summary_stats()/run_all_checks() surface numpy
    # scalars (Series.to_dict(), .sum(), etc.) in checks/summary_stats that
    # pydantic can't serialize on its own - sanitize the whole tree.
    return _json_safe_deep(report)



def _run_quality_analysis(
    content_bytes: bytes,
    file_extension: str,
    schema_dict: Optional[Dict[str, Any]],
    rules_dict: Optional[Dict[str, Any]],
    min_rows: int,
    label: str,
) -> Dict[str, Any]:
    """Parse dataset bytes and run the shared quality analysis.

    Single code path behind both POST /api/v1/analyze (multipart upload) and
    POST /api/v1/analyze/json (inline content), so the two routes cannot
    drift apart.

    Parameters:
        content_bytes: Raw dataset bytes.
        file_extension: Lower-case extension selecting the parser
            ('.csv', '.json', '.txt', '.tsv', '.xlsx', '.xls').
        schema_dict: Optional column-type map.
        rules_dict: Optional validation rules.
        min_rows: Minimum required row count.
        label: Name used in log lines (file name or "inline <format>").

    Returns:
        The report dict from _build_quality_report().

    Raises:
        HTTPException: 413 for content over the size limit, 400 for
            empty/unparseable content, 500 otherwise.
    """
    limit = _max_request_bytes()
    if len(content_bytes) > limit:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=_too_large_detail(limit)
        )

    fingerprint = _content_fingerprint(content_bytes)
    try:
        if not content_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )

        # Mirror web_app.py's file_uploader parsing (pandas, by extension) so
        # the engine sees the same DataFrame whether loaded via the UI or
        # posted directly to this endpoint. Excel (.xlsx/.xls) is dispatched
        # through mcp_server.DataLoader.load_excel - the same shared loader
        # web_app.py uses locally for Excel parsing/previewing - so CSV/API/
        # UI callers all get identical Excel-handling behavior (sheet
        # combination, size checks, etc.) from one place.
        try:
            if file_extension == '.csv':
                # nrows bounds memory: stop reading after max_rows + 1 rows
                df = pd.read_csv(
                    io.BytesIO(content_bytes), nrows=mcp_server.DataLoader.max_rows() + 1
                )
            elif file_extension == '.json':
                df = pd.read_json(io.BytesIO(content_bytes))
            elif file_extension in ('.xlsx', '.xls'):
                df = mcp_server.DataLoader.load_excel(content_bytes)
            else:  # .txt / .tsv
                df = pd.read_csv(
                    io.BytesIO(content_bytes), sep='\t',
                    nrows=mcp_server.DataLoader.max_rows() + 1,
                )
            mcp_server.DataLoader.check_dataframe_limits(df)
        except mcp_server.DataTooLargeError as e:
            # Message is built from our own limit names/numbers only.
            logger.warning("analysis_refused_too_large ext=%s %s", file_extension, fingerprint)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=str(e),
            )
        except Exception as e:
            # Library text can quote cell values: log only class + fingerprint.
            logger.warning(
                "analysis_parse_failed ext=%s %s error_class=%s",
                file_extension, fingerprint, type(e).__name__,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse the uploaded {file_extension} file. "
                       "Check that it is a valid file of that type."
            )

        # Run the shared validation engine - single source of truth for all
        # quality-check logic (row count, type, and range/allowed-value checks).
        pipeline = mcp_server.QualityPipeline(df, schema=schema_dict, rules=rules_dict)
        results = pipeline.run_all_checks(min_rows=min_rows)

        report = _build_quality_report(df, results)

        logger.info(
            "Analysis complete for %s: %d issues (%d critical, %d warnings)",
            label, report['summary']['issues_found'],
            report['summary']['critical_issues'], report['summary']['warnings'],
        )
        return report

    except HTTPException:
        raise
    except Exception as e:
        # Never log str(e): it may contain uploaded values.
        logger.error(
            "analysis_unexpected_error ext=%s %s error_class=%s\n%s",
            file_extension, fingerprint, type(e).__name__,
            "".join(traceback.format_tb(e.__traceback__)),  # frames only, no message
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while analysing the data."
        )

# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with standardized error format"""
    # Preserve any headers set on the raised HTTPException (e.g. the
    # WWW-Authenticate: ApiKey header verify_api_key attaches to 401
    # responses). JSONResponse defaults to no headers, so without this the
    # standardized-error-format wrapper silently drops them.
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail or "An error occurred",
            detail=str(exc),
            timestamp=datetime.now().isoformat(),
            request_id=_current_request_id(),
        ).dict(),
        headers=exc.headers,
    )


def _debug_errors_enabled() -> bool:
    """Detailed client errors: DEBUG=true AND a local (non-deployed) environment.

    In a deployed environment (ACA / APP_ENV not dev|test|local) DEBUG has no
    effect on what clients are told.
    """
    return os.getenv("DEBUG") == "true" and not is_deployed_env()


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with standardized error format"""
    request_id = _current_request_id()
    # Exception text can contain uploaded values: log the class and the frames only.
    logger.error(
        "unhandled_exception request_id=%s error_class=%s\n%s",
        request_id, type(exc).__name__, "".join(traceback.format_tb(exc.__traceback__)),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if _debug_errors_enabled() else "An unexpected error occurred",
            timestamp=datetime.now().isoformat(),
            request_id=request_id,
        ).dict(),
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle a stray ValueError with a generic 400 (no library text to the client)."""
    request_id = _current_request_id()
    logger.warning("value_error request_id=%s error_class=%s", request_id, type(exc).__name__)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="Invalid input",
            detail=str(exc) if _debug_errors_enabled() else "The request contained an invalid value.",
            timestamp=datetime.now().isoformat(),
            request_id=request_id,
        ).dict(),
    )

@app.exception_handler(RequestValidationError)
async def json_route_validation_handler(request: Request, exc: RequestValidationError):
    """Return the standardized error format for a bad /api/v1/analyze/json body.

    Scoped to that route so every other endpoint keeps FastAPI's default 422
    body. Only loc/msg are reported: the raw errors() also echo the caller's
    input back, which may be the dataset itself.
    """
    if request.url.path != ANALYZE_JSON_PATH:
        return await request_validation_exception_handler(request, exc)
    problems = "; ".join(
        f"{'.'.join(str(p) for p in err.get('loc', ()))}: {err.get('msg', 'invalid')}"
        for err in exc.errors()
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Invalid request body",
            detail=problems,
            timestamp=datetime.now().isoformat(),
            request_id=_current_request_id(),
        ).dict(),
    )

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - redirect to API docs"""
    return {
        "message": "Data Analyzer REST API",
        "version": __version__,
        "docs": "/api/v1/docs",
        "health": "/api/v1/health",
        "version_info": "/api/v1/version",
        "analyze": ANALYZE_PATH,
        "analyze_json": ANALYZE_JSON_PATH,
    }

@app.get("/api/v1/health", response_model=HealthResponse)
@limiter.limit("60/minute")
async def health_check(request: Request):
    """
    Health check endpoint

    Returns system status and availability of services.
    No authentication required.

    **Rate Limit**: 60 requests per minute
    """
    services_status = {
        "llm_client": llm_client is not None,
        "program_manager": program_manager is not None,
        "logic_validator": logic_validator is not None,
        "mcp_server": mcp_server is not None,
    }

    # Overall status is healthy if at least basic services are available
    overall_status = "healthy" if all([
        services_status["program_manager"],
        services_status["logic_validator"],
    ]) else "degraded"

    return HealthResponse(
        status=overall_status,
        version=__version__,
        timestamp=datetime.now().isoformat(),
        services=services_status,
    )

@app.get("/api/v1/version", response_model=VersionResponse)
@limiter.limit("60/minute")
async def get_version(request: Request):
    """
    Version endpoint (ST-360)

    Returns the package version from `src/__version__.py` - the same value
    `/api/v1/health` and the web UI report. No authentication required.

    **Rate Limit**: 60 requests per minute
    """
    return VersionResponse(name="data-analyzer", version=__version__)

# ============================================================================
# Data Analysis Endpoints
# ============================================================================

@app.post(
    ANALYZE_PATH,
    response_model=QualityAnalysisResponse,
    responses={413: {"model": ErrorResponse, "description": "Request larger than DATA_ANALYZER_MAX_UPLOAD_BYTES"}},
)
@limiter.limit("30/minute")
async def analyze_data(
    request: Request,
    data_file: UploadFile = File(..., description="Dataset file to analyze (CSV, JSON, tab-separated TXT, or Excel)"),
    # Python identifier is column_schema (not "schema") to avoid shadowing
    # BaseModel.schema() on FastAPI's auto-generated request body model; the
    # wire-level form field name stays "schema" via alias=.
    column_schema: Optional[str] = Form(
        None,
        alias="schema",
        description='Optional JSON-encoded column-type map, e.g. \'{"age": "int"}\'. '
                     'Supported types: int, float, str, bool, datetime.'
    ),
    rules: Optional[str] = Form(
        None,
        description='Optional JSON-encoded validation rules, e.g. \'{"age": {"min": 0, "max": 120}}\'. '
                     'Supports "min"/"max" for numeric columns and "allowed" for categorical columns.'
    ),
    min_rows: int = Form(1, ge=0, description="Minimum required row count"),
    api_key: str = Depends(verify_api_key)
):
    """
    Run rule-based data quality analysis on an uploaded dataset.

    This exposes the same validation engine (QualityPipeline/QualityChecker
    from mcp_server.py) that web_app.py's dashboard has always used - it is
    rule-based and does NOT call an LLM, so it works with no LLM configured.

    **Supported formats:** CSV (`.csv`), JSON (`.json`), tab-separated
    (`.txt`/`.tsv`), Excel (`.xlsx`/`.xls`) - mirrors the formats the
    Streamlit UI's file uploader already accepts.

    **Rate Limit**: 30 requests per minute (no LLM cost, so a higher limit
    than dictionary parsing is fine)

    **Authentication**: Requires valid API key via X-API-Key header

    **Parameters:**
    - `data_file`: Dataset file to analyze
    - `schema`: Optional JSON string mapping column names to expected types
      (`int`, `float`, `str`, `bool`, `datetime`)
    - `rules`: Optional JSON string mapping column names to validation rules
      (`min`/`max` for numeric columns, `allowed` for categorical columns)
    - `min_rows`: Minimum required row count (default 1)

    **Returns:**
    - `summary`: row/column counts, issue counts, completeness percentage
    - `issues`: one entry per violation (range/type/categorical/missing-value),
      each with column, row, value, severity, and a human-readable message
    - `recommendations`: suggested next steps based on the issues found
    - `quality_checks`: raw per-check results from QualityPipeline
    - `summary_stats`: dataset shape, dtypes, missing-value counts, numeric stats
    """
    if mcp_server is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Analysis engine not available (mcp_server failed to import)."
        )

    # Validate file format
    filename = data_file.filename or "uploaded"
    file_extension = Path(filename).suffix.lower()
    supported_extensions = ['.csv', '.json', '.txt', '.tsv', '.xlsx', '.xls']

    if file_extension not in supported_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: {file_extension}. Supported: {', '.join(supported_extensions)}"
        )

    # Parse optional JSON form fields
    schema_dict: Optional[Dict[str, Any]] = None
    rules_dict: Optional[Dict[str, Any]] = None

    if column_schema:
        try:
            schema_dict = json.loads(column_schema)
            if not isinstance(schema_dict, dict):
                raise ValueError("'schema' must be a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid 'schema' JSON: it must be a JSON object."
            )

    if rules:
        try:
            rules_dict = json.loads(rules)
            if not isinstance(rules_dict, dict):
                raise ValueError("'rules' must be a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid 'rules' JSON: it must be a JSON object."
            )

    # The file name is caller-controlled (may carry identifiers, CR/LF): log
    # only its extension and, after reading, size + hash.
    logger.info(
        "Analyzing uploaded %s file (schema=%s, rules=%s)",
        file_extension, 'yes' if schema_dict else 'no', 'yes' if rules_dict else 'no',
    )

    content_bytes = await data_file.read()
    # Parsing and checking are CPU-bound pandas work: run them on a worker
    # thread so the event loop (and /api/v1/health) stays responsive.
    return await run_in_threadpool(
        _run_quality_analysis,
        content_bytes, file_extension, schema_dict, rules_dict, min_rows,
        f"uploaded {file_extension}",
    )


# Parser extension for each "format" value accepted by /api/v1/analyze/json.
_JSON_ROUTE_EXTENSIONS = {"csv": ".csv", "tsv": ".tsv", "json": ".json"}


@app.post(
    ANALYZE_JSON_PATH,
    response_model=QualityAnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Empty, undecodable or unparseable data_content"},
        413: {"model": ErrorResponse, "description": "Request larger than DATA_ANALYZER_MAX_UPLOAD_BYTES"},
        422: {"model": ErrorResponse, "description": "Request body failed validation"},
    },
)
@limiter.limit("30/minute")
async def analyze_data_json(
    request: Request,
    body: AnalyzeJsonRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Run rule-based data quality analysis on a dataset sent inline as JSON (ST-359).

    Same analysis and same response as the multipart `POST /api/v1/analyze`,
    for programs that would rather post a JSON body than build a file
    upload. Rule-based: does NOT call an LLM.

    **Rate Limit**: 30 requests per minute

    **Authentication**: Requires valid API key via X-API-Key header

    **Body:**
    - `data_content`: CSV/TSV/JSON text, or base64 when `encoding` is `"base64"`
    - `format`: `csv` (default), `tsv` or `json`
    - `encoding`: `text` (default) or `base64`
    - `schema`: Optional object mapping column names to expected types
    - `rules`: Optional object mapping column names to validation rules
    - `min_rows`: Minimum required row count (default 1)

    **Errors:** 400 (bad content), 413 (over the size limit) and 422 (bad
    body) use the standard `ErrorResponse` JSON shape.
    """
    if mcp_server is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Analysis engine not available (mcp_server failed to import)."
        )

    label = f"inline {body.format}"
    logger.info(
        f"Analyzing {label} content (schema={'yes' if body.column_schema else 'no'}, "
        f"rules={'yes' if body.rules else 'no'})"
    )

    def _decode_and_analyze() -> Dict[str, Any]:
        if body.encoding == "base64":
            try:
                content_bytes = base64.b64decode(body.data_content, validate=True)
            except (binascii.Error, ValueError):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid base64 in 'data_content'."
                )
        else:
            content_bytes = body.data_content.encode("utf-8")
        return _run_quality_analysis(
            content_bytes,
            _JSON_ROUTE_EXTENSIONS[body.format],
            body.column_schema,
            body.rules,
            body.min_rows,
            label=label,
        )

    # Decoding + parsing are blocking: keep them off the event loop.
    return await run_in_threadpool(_decode_and_analyze)

# ============================================================================
# Dictionary Management Endpoints
# ============================================================================

@app.post(
    DICTIONARY_PARSE_PATH,
    response_model=ParseDictionaryResponse,
    responses={413: {"model": ErrorResponse, "description": "Request larger than DATA_ANALYZER_MAX_DICTIONARY_UPLOAD_BYTES"}},
)
@limiter.limit("5/minute")
async def parse_dictionary(
    request: Request,
    dictionary_file: UploadFile = File(..., description="Dictionary file (PDF, CSV, JSON, TXT)"),
    save_program: bool = Form(True, description="Whether to save as cached program"),
    program_name: Optional[str] = Form(None, description="Custom name override"),
    api_key: str = Depends(verify_api_key)
):
    """
    Parse data dictionary and generate validation program

    This endpoint accepts a data dictionary file, parses it using LLM to extract
    field definitions and validation rules, and optionally saves the resulting
    validation program to the database.

    **Supported formats:** PDF, CSV, JSON, TXT (REDCap CSV, FHIR JSON, generic formats)

    **Rate Limit**: 5 requests per minute (LLM calls are expensive)

    **Authentication**: Requires valid API key via X-API-Key header

    **Parameters:**
    - `dictionary_file`: Dictionary file to parse
    - `save_program`: Whether to save the generated program (default: True)
    - `program_name`: Optional custom name for the program (auto-generated if not provided)

    **Returns:**
    - Program ID, name, and metadata
    - Number of fields, rules, and logic rules extracted
    - Generated validation code
    - Field schema
    """
    start_time = datetime.now()

    # Validate the request before checking backend availability, so a bad
    # upload gets a 400 whether or not the LLM service is configured.
    # Validate file format
    filename = dictionary_file.filename or "uploaded"
    file_extension = Path(filename).suffix.lower()
    supported_extensions = ['.pdf', '.csv', '.json', '.txt']

    if file_extension not in supported_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: {file_extension}. Supported: {', '.join(supported_extensions)}"
        )

    # Validate service availability
    if not program_manager:
        logger.error("Program manager not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dictionary parsing service not available. LLM client may not be configured."
        )

    # File name is caller-controlled: log extension (+ size/hash below), not the name.
    logger.info("Parsing dictionary file (ext=%s, save_program=%s)", file_extension, save_program)

    try:
        # Read file content
        content_bytes = await dictionary_file.read()

        # Second line of defence behind MaxRequestSizeMiddleware: the LLM is
        # called roughly once per ~4.5k tokens of this text, so the input
        # handed to it must be bounded.
        dictionary_limit = _max_dictionary_request_bytes()
        if len(content_bytes) > dictionary_limit:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=_too_large_detail(dictionary_limit, "DATA_ANALYZER_MAX_DICTIONARY_UPLOAD_BYTES"),
            )
        fingerprint = _content_fingerprint(content_bytes)

        # Decode based on file type
        if file_extension == '.pdf':
            # TODO: Implement PDF parsing (PyPDF2 or pdfplumber)
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="PDF parsing not yet implemented. Please convert to CSV or JSON."
            )
        else:
            # Text-based formats
            try:
                dictionary_content = content_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # latin-1 maps every byte, so this fallback cannot fail
                dictionary_content = content_bytes.decode('latin-1')

        logger.info("Read dictionary: ext=%s %s chars=%d", file_extension, fingerprint, len(dictionary_content))

        dictionary_path = Path(filename)

        def _parse_and_save():
            # Blocking: LLM HTTP calls, tiktoken, SQLite. Runs on a worker
            # thread (llm_action's context is propagated by run_in_threadpool).
            with llm_action("api_parse_dictionary"):
                kwargs = {"program_name": program_name} if (program_name and save_program) else {}
                return program_manager.create_program_from_dictionary(
                    dictionary_content=dictionary_content,
                    dictionary_path=dictionary_path,
                    save=save_program,
                    **kwargs,
                )

        try:
            program = await run_in_threadpool(_parse_and_save)
        except ProgramSaveError as e:
            logger.error("dictionary_save_failed ext=%s %s error_class=%s", file_extension, fingerprint, type(e).__name__)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The dictionary was parsed but could not be saved, so no program was stored. "
                       "A program with the same name may already exist."
            )
        except RuntimeError as e:
            logger.error("dictionary_parse_failed ext=%s %s error_class=%s", file_extension, fingerprint, type(e).__name__)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Dictionary parsing failed."
            )

        # Calculate generation time
        generation_time = (datetime.now() - start_time).total_seconds()

        # Build response
        response = ParseDictionaryResponse(
            program_id=program.program_id,
            program_name=program.name,
            fields_extracted=program.num_fields,
            rules_extracted=program.num_basic_rules,
            logic_rules_extracted=program.num_logic_rules,
            generated_code=program.generated_code[:500] + "..." if len(program.generated_code) > 500 else program.generated_code,
            schema=program.schema,
            dictionary_format=program.dictionary_format,
            generation_time_seconds=generation_time,
            model_used=program.model_used
        )

        logger.info("Successfully parsed dictionary: program_id=%s (%s fields)", program.program_id, program.num_fields)
        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(
            "dictionary_unexpected_error ext=%s error_class=%s\n%s",
            file_extension, type(e).__name__, "".join(traceback.format_tb(e.__traceback__)),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while parsing the dictionary."
        )


@app.get("/api/v1/dictionary/{dict_id}", response_model=ProgramDetail)
@limiter.limit("30/minute")
async def get_dictionary(
    request: Request,
    dict_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get dictionary details by ID, name, or alias

    Retrieves a saved validation program (parsed dictionary) from the database.
    The program can be identified by:
    - Program ID (UUID)
    - Program name (e.g., "20241202-143022-ClinicalTrial")
    - Program alias (e.g., "johnDoesFav01")

    **Rate Limit**: 30 requests per minute

    **Authentication**: Requires valid API key via X-API-Key header

    **Parameters:**
    - `dict_id`: Program identifier (ID, name, or alias)

    **Returns:**
    - Complete program details including:
      - Metadata (creation info, usage statistics)
      - Field schema
      - Validation rules
      - Generated code
      - Conditional logic rules

    **Status Codes:**
    - `200`: Program found and returned
    - `404`: Program not found
    - `500`: Database error
    """
    safe_id = _log_safe(dict_id)  # URL-decoded caller text: strip control chars, bound length
    logger.info("Fetching dictionary: %s", safe_id)

    # Validate service availability
    if not program_manager:
        logger.error("Program manager not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Program management service not available"
        )

    try:
        # Load program from database
        # SQLite is blocking I/O: keep it off the event loop.
        program = await run_in_threadpool(program_manager.db.load_program, dict_id)

        if not program:
            logger.warning("Program not found: %s", safe_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Program not found."
            )

        # Check if program is deleted
        if program.status == "deleted":
            logger.warning("Attempted to retrieve deleted program: %s", safe_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Program has been deleted."
            )

        # Convert to API model
        program_detail = convert_validation_program_to_detail(program)

        logger.info("Successfully retrieved program: %s", program.program_id)
        return program_detail

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(
            "dictionary_lookup_error error_class=%s\n%s",
            type(e).__name__, "".join(traceback.format_tb(e.__traceback__)),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error."
        )

# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    logger.info("=" * 60)
    logger.info("Data Analyzer REST API starting up")
    logger.info(f"Version: {__version__}")
    logger.info(f"Environment: {os.getenv('APP_ENV', 'development')}")
    logger.info(f"Services initialized:")
    logger.info(f"  - LLM Client: {'✓' if llm_client else '✗'}")
    logger.info(f"  - Program Manager: {'✓' if program_manager else '✗'}")
    logger.info(f"  - Logic Validator: {'✓' if logic_validator else '✗'}")
    logger.info(f"  - MCP Server: {'✓' if mcp_server else '✗'}")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Data Analyzer REST API shutting down")
    # TODO: Add any cleanup logic here (close database connections, etc.)

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    # Default to no hot-reload: reload spawns a file-watching subprocess that
    # has no place in a deployed container and increases attack surface.
    # Opt in explicitly for local dev via API_RELOAD=true.
    reload = os.getenv("API_RELOAD", "false").lower() == "true"

    logger.info(f"Starting server at {host}:{port}")

    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
