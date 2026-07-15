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
    ReturnFormatEnum
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
