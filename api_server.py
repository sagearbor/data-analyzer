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

import io
import logging
import os
import secrets
import traceback
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
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
)

# Optional service modules - each is independently best-effort. A missing/broken
# module here degrades the corresponding feature (endpoints return 503) instead
# of taking down the whole server.
try:
    from src.llm_client import LLMDictionaryParser
except ImportError as e:
    logging.error(f"Import error (LLMDictionaryParser unavailable): {e}")
    LLMDictionaryParser = None

try:
    from src.program_manager import ProgramManager
except ImportError as e:
    logging.error(f"Import error (ProgramManager unavailable): {e}")
    ProgramManager = None

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
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
)

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Fail closed in production: missing credentials must abort startup, not
# silently disable authentication (see verify_api_key dev-mode fallback).
if os.getenv("APP_ENV", "dev") == "prod" and not API_KEY:
    raise RuntimeError(
        "APP_ENV=prod but DATA_ANALYZER_API_KEY is not set. "
        "Refusing to start with authentication disabled. "
        "Set DATA_ANALYZER_API_KEY (and DATA_ANALYZER_ADMIN_PASSWORD for "
        "admin endpoints) in the container environment."
    )

# Security scheme for API key
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
admin_password_header = APIKeyHeader(name="X-Admin-Password", auto_error=False)

# Authentication dependency functions
async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)) -> str:
    """
    Verify API key from X-API-Key header

    Args:
        api_key: API key from request header

    Returns:
        The validated API key

    Raises:
        HTTPException: 401 if API key is missing, 403 if invalid
    """
    # Check if API key authentication is enabled
    if not API_KEY:
        logger.warning("API_KEY not configured - authentication disabled")
        return "unauthenticated"

    # Check if API key is provided
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide valid credentials.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Validate API key (constant-time comparison to avoid timing side-channels)
    if not secrets.compare_digest(api_key, API_KEY):
        # Log the failed attempt (without logging the actual key)
        logger.warning(f"Invalid API key attempt from request")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid credentials provided.",
        )

    return api_key

async def verify_admin_password(admin_password: Optional[str] = Depends(admin_password_header)) -> str:
    """
    Verify admin password from X-Admin-Password header

    Args:
        admin_password: Admin password from request header

    Returns:
        The validated admin password

    Raises:
        HTTPException: 401 if password is missing, 403 if invalid
    """
    # Check if admin password authentication is enabled
    if not ADMIN_PASSWORD:
        logger.warning("ADMIN_PASSWORD not configured - admin authentication disabled")
        return "unauthenticated"

    # Check if admin password is provided
    if not admin_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin authentication required. Provide valid credentials.",
            headers={"WWW-Authenticate": "AdminPassword"},
        )

    # Validate admin password (constant-time comparison to avoid timing side-channels)
    if not secrets.compare_digest(admin_password, ADMIN_PASSWORD):
        # Log the failed attempt
        logger.warning(f"Invalid admin password attempt from request")
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
        "issues_found": len(issues),
        "critical_issues": sum(1 for i in issues if i.get('severity') == 'error'),
        "warnings": sum(1 for i in issues if i.get('severity') == 'warning'),
        "data_types": results.get('summary_stats', {}).get('dtypes', {col: str(df[col].dtype) for col in df.columns}),
        "completeness": completeness
    }

    report = {
        "summary": summary,
        "issues": issues,
        "recommendations": _generate_quality_recommendations(issues),
        "quality_checks": results.get('checks', {}),
        "summary_stats": results.get('summary_stats', {})
    }
    # QualityChecker.get_summary_stats()/run_all_checks() surface numpy
    # scalars (Series.to_dict(), .sum(), etc.) in checks/summary_stats that
    # pydantic can't serialize on its own - sanitize the whole tree.
    return _json_safe_deep(report)

# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with standardized error format"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail or "An error occurred",
            detail=str(exc),
            timestamp=datetime.now().isoformat(),
            request_id=request.headers.get("X-Request-ID"),
        ).dict(),
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with standardized error format"""
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if os.getenv("DEBUG") == "true" else "An unexpected error occurred",
            timestamp=datetime.now().isoformat(),
            request_id=request.headers.get("X-Request-ID"),
        ).dict(),
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError with bad request status"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="Invalid input",
            detail=str(exc),
            timestamp=datetime.now().isoformat(),
            request_id=request.headers.get("X-Request-ID"),
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
        "version": "1.0.0",
        "docs": "/api/v1/docs",
        "health": "/api/v1/health",
        "analyze": "/api/v1/analyze",
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
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        services=services_status,
    )

# ============================================================================
# Data Analysis Endpoints
# ============================================================================

@app.post("/api/v1/analyze", response_model=QualityAnalysisResponse)
@limiter.limit("30/minute")
async def analyze_data(
    request: Request,
    data_file: UploadFile = File(..., description="Dataset file to analyze (CSV, JSON, or tab-separated TXT)"),
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
    (`.txt`/`.tsv`) - mirrors the formats the Streamlit UI's file uploader
    already accepts.

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
    supported_extensions = ['.csv', '.json', '.txt', '.tsv']

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
                detail=f"Invalid 'schema' JSON: {str(e)}"
            )

    if rules:
        try:
            rules_dict = json.loads(rules)
            if not isinstance(rules_dict, dict):
                raise ValueError("'rules' must be a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid 'rules' JSON: {str(e)}"
            )

    logger.info(f"Analyzing data file: {filename} (schema={'yes' if schema_dict else 'no'}, rules={'yes' if rules_dict else 'no'})")

    try:
        content_bytes = await data_file.read()
        if not content_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )

        # Mirror web_app.py's file_uploader parsing (pandas, by extension) so
        # the engine sees the same DataFrame whether loaded via the UI or
        # posted directly to this endpoint.
        try:
            if file_extension == '.csv':
                df = pd.read_csv(io.BytesIO(content_bytes))
            elif file_extension == '.json':
                df = pd.read_json(io.BytesIO(content_bytes))
            else:  # .txt / .tsv
                df = pd.read_csv(io.BytesIO(content_bytes), sep='\t')
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse {file_extension} file: {str(e)}"
            )

        # Run the shared validation engine - single source of truth for all
        # quality-check logic (row count, type, and range/allowed-value checks).
        pipeline = mcp_server.QualityPipeline(df, schema=schema_dict, rules=rules_dict)
        results = pipeline.run_all_checks(min_rows=min_rows)

        report = _build_quality_report(df, results)

        logger.info(
            f"Analysis complete for {filename}: {report['summary']['issues_found']} issues "
            f"({report['summary']['critical_issues']} critical, {report['summary']['warnings']} warnings)"
        )
        return report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error analyzing data: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )

# ============================================================================
# Dictionary Management Endpoints
# ============================================================================

@app.post("/api/v1/dictionary/parse", response_model=ParseDictionaryResponse)
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

    # Validate service availability
    if not program_manager:
        logger.error("Program manager not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dictionary parsing service not available. LLM client may not be configured."
        )

    # Validate file format
    filename = dictionary_file.filename or "uploaded"
    file_extension = Path(filename).suffix.lower()
    supported_extensions = ['.pdf', '.csv', '.json', '.txt']

    if file_extension not in supported_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: {file_extension}. Supported: {', '.join(supported_extensions)}"
        )

    logger.info(f"Parsing dictionary file: {filename} (save_program={save_program})")

    try:
        # Read file content
        content_bytes = await dictionary_file.read()

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
                # Try latin-1 as fallback
                try:
                    dictionary_content = content_bytes.decode('latin-1')
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to decode file: {str(e)}"
                    )

        logger.info(f"Read {len(dictionary_content)} characters from {filename}")

        # Parse dictionary and create program
        dictionary_path = Path(filename)

        try:
            program = program_manager.create_program_from_dictionary(
                dictionary_content=dictionary_content,
                dictionary_path=dictionary_path,
                save=save_program
            )
        except RuntimeError as e:
            logger.error(f"Dictionary parsing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Dictionary parsing failed: {str(e)}"
            )

        # Override name if provided
        if program_name and save_program:
            old_name = program.name
            program.name = program_name
            # Update in database
            try:
                program_manager.db.save_program(program)
                logger.info(f"Updated program name from {old_name} to {program_name}")
            except Exception as e:
                logger.warning(f"Failed to update program name: {e}")

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

        logger.info(f"Successfully parsed dictionary: {program.name} ({program.num_fields} fields)")
        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error parsing dictionary: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
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
    logger.info(f"Fetching dictionary: {dict_id}")

    # Validate service availability
    if not program_manager:
        logger.error("Program manager not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Program management service not available"
        )

    try:
        # Load program from database
        program = program_manager.db.load_program(dict_id)

        if not program:
            logger.warning(f"Program not found: {dict_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Program not found: {dict_id}"
            )

        # Check if program is deleted
        if program.status == "deleted":
            logger.warning(f"Attempted to retrieve deleted program: {dict_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Program '{program.name}' has been deleted"
            )

        # Convert to API model
        program_detail = convert_validation_program_to_detail(program)

        logger.info(f"Successfully retrieved program: {program.name}")
        return program_detail

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error retrieving program: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )

# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    logger.info("=" * 60)
    logger.info("Data Analyzer REST API starting up")
    logger.info(f"Version: 1.0.0")
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
