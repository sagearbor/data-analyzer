"""
Pydantic Models for REST API Request/Response Validation

This module provides comprehensive Pydantic models for the data-analyzer REST API:
- Analyze endpoints: /api/v1/analyze, /api/v1/analyze/with-program
- Dictionary management: /api/v1/dictionary/parse
- Program management: /api/v1/programs/*
- System health: /api/v1/health

All models include:
- Type hints and Field() descriptions
- Validation rules (min/max lengths, allowed values)
- Example values for OpenAPI schema generation
- Alignment with ValidationProgram dataclass from program_cache.py
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from enum import Enum


# =============================================================================
# ENUMS FOR VALIDATION
# =============================================================================

class DataFormatEnum(str, Enum):
    """Supported data file formats"""
    csv = "csv"
    json = "json"
    excel = "excel"
    parquet = "parquet"


class ReturnFormatEnum(str, Enum):
    """Supported return formats for analysis results"""
    json = "json"
    html = "html"
    excel = "excel"


class ProgramStatusEnum(str, Enum):
    """Program status values"""
    active = "active"
    deleted = "deleted"
    all = "all"


class SeverityEnum(str, Enum):
    """Issue severity levels"""
    error = "error"
    warning = "warning"
    info = "info"


# =============================================================================
# ANALYSIS MODELS
# =============================================================================

class AnalyzeOptions(BaseModel):
    """
    Options for data analysis requests.

    Attributes:
        format: Data file format (csv, json, excel, parquet)
        validate_logic: Whether to run conditional logic validation
        return_format: Format for analysis results (json, html, excel)
    """
    format: DataFormatEnum = Field(
        default=DataFormatEnum.csv,
        description="Format of the data file to analyze"
    )
    validate_logic: bool = Field(
        default=True,
        description="Whether to validate conditional logic rules"
    )
    return_format: ReturnFormatEnum = Field(
        default=ReturnFormatEnum.json,
        description="Format for the analysis results"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "format": "csv",
                "validate_logic": True,
                "return_format": "json"
            }
        }
    }


class FieldViolation(BaseModel):
    """
    Represents a field-level validation violation.

    Attributes:
        field_name: Name of the field with violation
        row_index: Row number where violation occurred (0-indexed)
        violation_type: Type of violation (e.g., "type_mismatch", "range_violation")
        expected: Expected value or constraint
        actual: Actual value that violated the constraint
        severity: Severity level (error, warning, info)
    """
    field_name: str = Field(description="Name of the field")
    row_index: int = Field(ge=0, description="Row number (0-indexed)")
    violation_type: str = Field(description="Type of violation")
    expected: str = Field(description="Expected value or constraint")
    actual: Any = Field(description="Actual value that violated constraint")
    severity: SeverityEnum = Field(default=SeverityEnum.error, description="Severity level")

    model_config = {
        "json_schema_extra": {
            "example": {
                "field_name": "age",
                "row_index": 42,
                "violation_type": "range_violation",
                "expected": "0 <= age <= 120",
                "actual": 150,
                "severity": "error"
            }
        }
    }


class LogicViolation(BaseModel):
    """
    Represents a conditional logic rule violation.

    Aligns with LogicViolation dataclass from logic_engine.py.

    Attributes:
        rule_id: ID of the rule that was violated
        rule_description: Human-readable description
        row_index: Row number where violation occurred (0-indexed)
        affected_fields: Field names involved in the violation
        actual_values: Actual values of the fields
        expected_behavior: What was expected according to the rule
        severity: Severity level (error, warning)
    """
    rule_id: str = Field(description="Unique rule identifier")
    rule_description: str = Field(description="Human-readable rule description")
    row_index: int = Field(ge=0, description="Row number (0-indexed)")
    affected_fields: List[str] = Field(description="Fields involved in violation")
    actual_values: Dict[str, Any] = Field(description="Actual field values")
    expected_behavior: str = Field(description="Expected behavior per rule")
    severity: SeverityEnum = Field(default=SeverityEnum.error, description="Severity level")

    model_config = {
        "json_schema_extra": {
            "example": {
                "rule_id": "skip_if_001",
                "rule_description": "If gender=1, then pregnancy_status should be blank",
                "row_index": 15,
                "affected_fields": ["gender", "pregnancy_status"],
                "actual_values": {"gender": 1, "pregnancy_status": "Yes"},
                "expected_behavior": "pregnancy_status should be blank when gender=1",
                "severity": "error"
            }
        }
    }


class AnalysisSummary(BaseModel):
    """
    Summary statistics for analysis results.

    Attributes:
        total_rows: Number of rows analyzed
        total_columns: Number of columns in dataset
        issues_found: Total number of field-level issues
        logic_violations: Total number of logic rule violations
        execution_time_seconds: Time taken for analysis
    """
    total_rows: int = Field(ge=0, description="Number of rows analyzed")
    total_columns: int = Field(ge=0, description="Number of columns in dataset")
    issues_found: int = Field(ge=0, description="Total field-level issues")
    logic_violations: int = Field(ge=0, description="Total logic violations")
    execution_time_seconds: float = Field(ge=0.0, description="Analysis execution time")

    model_config = {
        "json_schema_extra": {
            "example": {
                "total_rows": 1000,
                "total_columns": 25,
                "issues_found": 42,
                "logic_violations": 8,
                "execution_time_seconds": 2.5
            }
        }
    }


class AnalyzeResponse(BaseModel):
    """
    Response for analysis requests.

    Returned by:
    - POST /api/v1/analyze
    - POST /api/v1/analyze/with-program

    Attributes:
        analysis_id: Unique identifier for this analysis
        timestamp: When the analysis was performed
        summary: Summary statistics
        field_violations: List of field-level violations
        logic_violations: List of conditional logic violations
        recommendations: Suggestions for data quality improvement
        program_used: Program ID/name if cached program was used
    """
    analysis_id: str = Field(description="Unique analysis identifier (UUID)")
    timestamp: datetime = Field(description="Analysis timestamp")
    summary: AnalysisSummary = Field(description="Summary statistics")
    field_violations: List[FieldViolation] = Field(
        default=[],
        description="Field-level validation violations"
    )
    logic_violations: List[LogicViolation] = Field(
        default=[],
        description="Conditional logic violations"
    )
    recommendations: List[str] = Field(
        default=[],
        description="Data quality improvement suggestions"
    )
    program_used: Optional[str] = Field(
        default=None,
        description="Program ID or name if cached program was used"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "analysis_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "timestamp": "2024-12-02T14:30:00Z",
                "summary": {
                    "total_rows": 1000,
                    "total_columns": 25,
                    "issues_found": 42,
                    "logic_violations": 8,
                    "execution_time_seconds": 2.5
                },
                "field_violations": [],
                "logic_violations": [],
                "recommendations": [
                    "Consider standardizing date formats in 'enrollment_date' field",
                    "Review missing values in 'email' field (5% missing)"
                ],
                "program_used": "20241202-143022-ClinicalTrial"
            }
        }
    }


class AnalyzeWithProgramRequest(BaseModel):
    """
    Request for analyzing data with a cached validation program.

    Used by: POST /api/v1/analyze/with-program

    Attributes:
        program: Program name, ID, or alias
        options: Analysis options
    """
    program: str = Field(
        min_length=1,
        max_length=255,
        description="Program name, ID, or alias"
    )
    options: AnalyzeOptions = Field(
        default=AnalyzeOptions(),
        description="Analysis options"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "program": "johnDoesFav01",
                "options": {
                    "format": "csv",
                    "validate_logic": True,
                    "return_format": "json"
                }
            }
        }
    }


# =============================================================================
# DICTIONARY PARSING MODELS
# =============================================================================

class ParseDictionaryRequest(BaseModel):
    """
    Request for parsing a data dictionary.

    Used by: POST /api/v1/dictionary/parse

    Attributes:
        save_program: Whether to save the generated validation program
        program_name: Optional custom name for the program
        dictionary_format: Optional format hint (e.g., "REDCap CSV", "FHIR JSON")
    """
    save_program: bool = Field(
        default=True,
        description="Whether to cache the generated validation program"
    )
    program_name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Custom name for the program (auto-generated if not provided)"
    )
    dictionary_format: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Dictionary format hint (e.g., 'REDCap CSV', 'FHIR JSON')"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "save_program": True,
                "program_name": "ClinicalTrial_v1",
                "dictionary_format": "REDCap CSV"
            }
        }
    }


class ParseDictionaryResponse(BaseModel):
    """
    Response for dictionary parsing request.

    Used by: POST /api/v1/dictionary/parse

    Attributes:
        program_id: Unique program identifier (UUID)
        program_name: Auto-generated or custom name
        fields_extracted: Number of fields extracted from dictionary
        rules_extracted: Number of basic validation rules extracted
        logic_rules_extracted: Number of conditional logic rules extracted
        generated_code: Python validation code (truncated in response)
        field_schema: Field definitions extracted from dictionary
        dictionary_format: Detected format of the dictionary
        generation_time_seconds: Time taken to generate validation code
        model_used: LLM model used for code generation
    """
    program_id: str = Field(description="Unique program identifier (UUID)")
    program_name: str = Field(description="Program name")
    fields_extracted: int = Field(ge=0, description="Number of fields extracted")
    rules_extracted: int = Field(ge=0, description="Number of basic rules")
    logic_rules_extracted: int = Field(ge=0, description="Number of logic rules")
    generated_code: str = Field(description="Generated Python validation code (may be truncated)")
    field_schema: Dict[str, Any] = Field(description="Field definitions", alias="schema")
    dictionary_format: str = Field(description="Detected dictionary format")
    generation_time_seconds: float = Field(ge=0.0, description="Generation time")
    model_used: str = Field(description="LLM model used")

    model_config = {
        "json_schema_extra": {
            "example": {
                "program_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "program_name": "20241202-143022-ClinicalTrial",
                "fields_extracted": 45,
                "rules_extracted": 23,
                "logic_rules_extracted": 8,
                "generated_code": "def validate_logic_rules(df):\n    violations = []\n    ...",
                "schema": {
                    "patient_id": {"type": "int", "required": True},
                    "enrollment_date": {"type": "datetime", "required": True}
                },
                "dictionary_format": "REDCap CSV",
                "generation_time_seconds": 3.5,
                "model_used": "gpt-5-nano"
            }
        },
        "populate_by_name": True,
        "by_alias": True
    }


# =============================================================================
# PROGRAM MANAGEMENT MODELS
# =============================================================================

class ProgramSummary(BaseModel):
    """
    Summary information for program listings.

    Used by: GET /api/v1/programs

    Attributes:
        program_id: Unique program identifier
        name: Program name
        aliases: User-friendly aliases
        created_at: Creation timestamp
        last_used: Last execution timestamp
        use_count: Number of times executed
        num_fields: Number of fields in schema
        num_logic_rules: Number of conditional logic rules
        dictionary_source: Original dictionary filename
        status: Program status (active/deleted)
    """
    program_id: str = Field(description="Unique program identifier")
    name: str = Field(description="Program name")
    aliases: List[str] = Field(default=[], description="User-friendly aliases")
    created_at: datetime = Field(description="Creation timestamp")
    last_used: Optional[datetime] = Field(default=None, description="Last execution timestamp")
    use_count: int = Field(ge=0, description="Execution count")
    num_fields: int = Field(ge=0, description="Number of fields")
    num_logic_rules: int = Field(ge=0, description="Number of logic rules")
    dictionary_source: str = Field(description="Original dictionary filename")
    status: ProgramStatusEnum = Field(description="Program status")

    model_config = {
        "json_schema_extra": {
            "example": {
                "program_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "name": "20241202-143022-ClinicalTrial",
                "aliases": ["johnDoesFav01", "clinicalV1"],
                "created_at": "2024-12-02T14:30:22Z",
                "last_used": "2024-12-02T16:45:00Z",
                "use_count": 15,
                "num_fields": 45,
                "num_logic_rules": 8,
                "dictionary_source": "clinical_trial.csv",
                "status": "active"
            }
        }
    }


class ProgramDetail(BaseModel):
    """
    Detailed program information.

    Used by: GET /api/v1/programs/{id_or_alias}

    Attributes:
        program_id: Unique program identifier
        name: Program name
        aliases: User-friendly aliases
        dictionary_source: Original dictionary filename
        dictionary_format: Dictionary format (e.g., "REDCap CSV")
        created_by: Username of creator
        created_at: Creation timestamp
        last_used: Last execution timestamp
        use_count: Number of times executed
        model_used: LLM model used for generation
        generation_time_seconds: Time taken to generate code
        num_fields: Number of fields in schema
        num_basic_rules: Number of basic validation rules
        num_logic_rules: Number of conditional logic rules
        generated_code: Full Python validation code
        field_schema: Field definitions
        conditional_rules: Conditional logic rules
        status: Program status (active/deleted)
        version: Version number
    """
    program_id: str = Field(description="Unique program identifier")
    name: str = Field(description="Program name")
    aliases: List[str] = Field(default=[], description="User-friendly aliases")
    dictionary_source: str = Field(description="Original dictionary filename")
    dictionary_format: str = Field(description="Dictionary format")
    created_by: str = Field(description="Username of creator")
    created_at: datetime = Field(description="Creation timestamp")
    last_used: Optional[datetime] = Field(default=None, description="Last execution timestamp")
    use_count: int = Field(ge=0, description="Execution count")
    model_used: str = Field(description="LLM model used")
    generation_time_seconds: float = Field(ge=0.0, description="Generation time")
    num_fields: int = Field(ge=0, description="Number of fields")
    num_basic_rules: int = Field(ge=0, description="Number of basic rules")
    num_logic_rules: int = Field(ge=0, description="Number of logic rules")
    generated_code: str = Field(description="Generated Python validation code")
    field_schema: Dict[str, Any] = Field(description="Field definitions", alias="schema")
    conditional_rules: List[Dict[str, Any]] = Field(description="Conditional logic rules")
    status: ProgramStatusEnum = Field(description="Program status")
    version: int = Field(ge=1, description="Version number")

    model_config = {
        "json_schema_extra": {
            "example": {
                "program_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "name": "20241202-143022-ClinicalTrial",
                "aliases": ["johnDoesFav01", "clinicalV1"],
                "dictionary_source": "clinical_trial.csv",
                "dictionary_format": "REDCap CSV",
                "created_by": "john.doe",
                "created_at": "2024-12-02T14:30:22Z",
                "last_used": "2024-12-02T16:45:00Z",
                "use_count": 15,
                "model_used": "gpt-5-nano",
                "generation_time_seconds": 3.5,
                "num_fields": 45,
                "num_basic_rules": 23,
                "num_logic_rules": 8,
                "generated_code": "def validate_logic_rules(df):\n    violations = []\n    ...",
                "schema": {},
                "conditional_rules": [],
                "status": "active",
                "version": 1
            }
        },
        "populate_by_name": True,
        "by_alias": True
    }


class ProgramListResponse(BaseModel):
    """
    Response for program listing request.

    Used by: GET /api/v1/programs

    Attributes:
        total: Total number of matching programs
        programs: List of program summaries
        limit: Pagination limit applied
        offset: Pagination offset applied
    """
    total: int = Field(ge=0, description="Total matching programs")
    programs: List[ProgramSummary] = Field(description="Program summaries")
    limit: int = Field(ge=1, description="Pagination limit")
    offset: int = Field(ge=0, description="Pagination offset")

    model_config = {
        "json_schema_extra": {
            "example": {
                "total": 42,
                "programs": [],
                "limit": 20,
                "offset": 0
            }
        }
    }


class CreateAliasRequest(BaseModel):
    """
    Request to create an alias for a program.

    Used by: POST /api/v1/programs/{id}/alias

    Attributes:
        alias: Alias string (globally unique, alphanumeric plus hyphens/underscores)
    """
    alias: str = Field(
        min_length=1,
        max_length=50,
        description="Globally unique alias (alphanumeric, hyphens, underscores)"
    )

    @field_validator('alias')
    @classmethod
    def validate_alias(cls, v: str) -> str:
        """Validate alias format"""
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Alias must contain only alphanumeric characters, hyphens, and underscores')
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "alias": "johnDoesFav01"
            }
        }
    }


class CreateAliasResponse(BaseModel):
    """
    Response for alias creation.

    Used by: POST /api/v1/programs/{id}/alias

    Attributes:
        success: Whether alias was created successfully
        alias: The created alias
        program_id: Program ID the alias points to
        message: Optional message (e.g., error details)
    """
    success: bool = Field(description="Whether operation succeeded")
    alias: str = Field(description="The alias that was created")
    program_id: str = Field(description="Program ID")
    message: Optional[str] = Field(default=None, description="Optional message")

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "alias": "johnDoesFav01",
                "program_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "message": "Alias created successfully"
            }
        }
    }


class DeleteProgramRequest(BaseModel):
    """
    Request to delete a program (admin only).

    Used by: DELETE /api/v1/programs/{id}

    Attributes:
        reason: Reason for deletion (required for audit trail)
    """
    reason: str = Field(
        min_length=10,
        max_length=500,
        description="Reason for deletion (required for audit trail)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "reason": "Contains errors in validation logic that cause false positives"
            }
        }
    }


class DeleteProgramResponse(BaseModel):
    """
    Response for program deletion.

    Used by: DELETE /api/v1/programs/{id}

    Attributes:
        success: Whether deletion succeeded
        program_id: ID of deleted program
        deleted_at: Timestamp of deletion
        message: Optional message
    """
    success: bool = Field(description="Whether operation succeeded")
    program_id: str = Field(description="Program ID")
    deleted_at: datetime = Field(description="Deletion timestamp")
    message: Optional[str] = Field(default=None, description="Optional message")

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "program_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "deleted_at": "2024-12-02T15:00:00Z",
                "message": "Program soft-deleted successfully"
            }
        }
    }


class RestoreProgramResponse(BaseModel):
    """
    Response for program restoration.

    Used by: POST /api/v1/programs/{id}/restore

    Attributes:
        success: Whether restoration succeeded
        program_id: ID of restored program
        restored_at: Timestamp of restoration
        message: Optional message
    """
    success: bool = Field(description="Whether operation succeeded")
    program_id: str = Field(description="Program ID")
    restored_at: datetime = Field(description="Restoration timestamp")
    message: Optional[str] = Field(default=None, description="Optional message")

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "program_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "restored_at": "2024-12-02T15:30:00Z",
                "message": "Program restored successfully"
            }
        }
    }


# =============================================================================
# ERROR HANDLING MODELS
# =============================================================================

class ErrorDetail(BaseModel):
    """
    Detailed error information.

    Attributes:
        field: Field name that caused error (if applicable)
        message: Error message
        type: Error type (e.g., "validation_error", "not_found")
    """
    field: Optional[str] = Field(default=None, description="Field name")
    message: str = Field(description="Error message")
    type: str = Field(description="Error type")

    model_config = {
        "json_schema_extra": {
            "example": {
                "field": "alias",
                "message": "Alias already exists",
                "type": "validation_error"
            }
        }
    }


class ErrorResponse(BaseModel):
    """
    Standardized error response format.

    Used by all endpoints when errors occur.

    Attributes:
        error: Human-readable error message
        error_code: Machine-readable error code (e.g., "PROGRAM_NOT_FOUND")
        details: Optional list of detailed error information
        timestamp: When the error occurred
        request_id: Optional request identifier for tracing
    """
    error: str = Field(description="Human-readable error message")
    error_code: str = Field(description="Machine-readable error code")
    details: Optional[List[ErrorDetail]] = Field(
        default=None,
        description="Detailed error information"
    )
    timestamp: datetime = Field(description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracing")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "Program not found",
                "error_code": "PROGRAM_NOT_FOUND",
                "details": [
                    {
                        "field": "program",
                        "message": "No program found with ID or alias 'nonexistent'",
                        "type": "not_found"
                    }
                ],
                "timestamp": "2024-12-02T14:30:00Z",
                "request_id": "req_a1b2c3d4"
            }
        }
    }


# =============================================================================
# SYSTEM HEALTH MODELS
# =============================================================================

class HealthCheckResponse(BaseModel):
    """
    Health check response.

    Used by: GET /api/v1/health

    Attributes:
        status: Overall health status (healthy, degraded, unhealthy)
        version: API version
        uptime_seconds: Server uptime in seconds
        database: Database connection status
        llm_service: LLM service connection status
        timestamp: Health check timestamp
        details: Optional detailed health information
    """
    status: str = Field(description="Overall health status")
    version: str = Field(description="API version")
    uptime_seconds: float = Field(ge=0.0, description="Server uptime")
    database: str = Field(description="Database status (connected/disconnected)")
    llm_service: str = Field(description="LLM service status (connected/disconnected)")
    timestamp: datetime = Field(description="Health check timestamp")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional health information"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "uptime_seconds": 3600,
                "database": "connected",
                "llm_service": "connected",
                "timestamp": "2024-12-02T14:30:00Z",
                "details": {
                    "database_path": "~/.data_analyzer/programs.db",
                    "total_programs": 42,
                    "total_executions": 1250
                }
            }
        }
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def convert_validation_program_to_detail(program: Any) -> ProgramDetail:
    """
    Convert ValidationProgram dataclass to ProgramDetail Pydantic model.

    Args:
        program: ValidationProgram instance from program_cache.py

    Returns:
        ProgramDetail instance for API response
    """
    return ProgramDetail(
        program_id=program.program_id,
        name=program.name,
        aliases=program.aliases,
        dictionary_source=program.dictionary_source,
        dictionary_format=program.dictionary_format,
        created_by=program.created_by,
        created_at=program.created_at,
        last_used=program.last_used,
        use_count=program.use_count,
        model_used=program.model_used,
        generation_time_seconds=program.generation_time_seconds,
        num_fields=program.num_fields,
        num_basic_rules=program.num_basic_rules,
        num_logic_rules=program.num_logic_rules,
        generated_code=program.generated_code,
        field_schema=program.schema,  # Note: field_schema maps to program.schema
        conditional_rules=program.conditional_rules,
        status=program.status,
        version=program.version
    )


def convert_validation_program_to_summary(program: Any) -> ProgramSummary:
    """
    Convert ValidationProgram dataclass to ProgramSummary Pydantic model.

    Args:
        program: ValidationProgram instance from program_cache.py

    Returns:
        ProgramSummary instance for API response
    """
    return ProgramSummary(
        program_id=program.program_id,
        name=program.name,
        aliases=program.aliases,
        created_at=program.created_at,
        last_used=program.last_used,
        use_count=program.use_count,
        num_fields=program.num_fields,
        num_logic_rules=program.num_logic_rules,
        dictionary_source=program.dictionary_source,
        status=program.status
    )
