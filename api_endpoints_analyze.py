"""
Data Analysis Endpoints for api_server.py

These two endpoints should be added to api_server.py in the "Data Analysis Endpoints" section:
1. POST /api/v1/analyze - Analyze data with optional dictionary
2. POST /api/v1/analyze/with-program - Analyze using cached validation program

Integration Instructions:
1. Add these endpoints to api_server.py after the health check endpoint
2. Ensure all imports from src.api_models are already present in api_server.py
3. The endpoints use verify_api_key dependency which is already defined
"""

from typing import Optional
from fastapi import UploadFile, File, Form, Depends, HTTPException, Request, status
from slowapi import limiter
import time
import uuid as uuid_lib
from datetime import datetime
from pathlib import Path
import tempfile
import pandas as pd
import traceback
import logging

# These imports should already be in api_server.py:
# from src.api_models import (
#     AnalyzeResponse, AnalysisSummary, FieldViolation, LogicViolation,
#     SeverityEnum, DataFormatEnum, ReturnFormatEnum
# )
# import mcp_server
# from program_manager import program_manager
# from logic_validator import logic_validator

logger = logging.getLogger(__name__)


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
@limiter.limit("10/minute")
async def analyze_data(
    request: Request,
    data_file: UploadFile = File(..., description="Data file to analyze (CSV, JSON, Excel, Parquet)"),
    dictionary_file: Optional[UploadFile] = File(None, description="Optional data dictionary file"),
    data_format: DataFormatEnum = Form(DataFormatEnum.csv, description="Data file format"),
    validate_logic: bool = Form(True, description="Whether to validate conditional logic rules"),
    return_format: ReturnFormatEnum = Form(ReturnFormatEnum.json, description="Format for results"),
    api_key: str = Depends(verify_api_key)
):
    """
    Analyze data file with optional data dictionary

    **Authentication**: Requires API key via X-API-Key header

    **Rate Limit**: 10 requests per minute

    **Workflow**:
    1. Load and validate the data file
    2. If dictionary provided: parse it and create/find validation program
    3. Run QualityPipeline for basic checks (types, ranges, etc.)
    4. If validate_logic=True and program available: run logic validation
    5. Return comprehensive analysis results

    **File Size Limits**:
    - Data file: 50 MB max
    - Dictionary file: 10 MB max

    **Supported Data Formats**:
    - CSV (with encoding auto-detection)
    - JSON (with nested structure flattening)
    - Excel (.xlsx, .xls)
    - Parquet

    **Response**: AnalyzeResponse with field violations, logic violations, and recommendations
    """
    start_time = time.time()
    analysis_id = str(uuid_lib.uuid4())

    logger.info(f"[{analysis_id}] Starting analysis request for file: {data_file.filename}")

    # Validate services are available
    if not mcp_server:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data analysis service unavailable (mcp_server module not loaded)"
        )

    try:
        # -------------------------------------------------------------------------
        # Step 1: Load and validate data file
        # -------------------------------------------------------------------------

        # Check file size (50 MB max for data files)
        data_content = await data_file.read()
        if len(data_content) > 50 * 1024 * 1024:  # 50 MB
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Data file too large. Maximum size is 50 MB."
            )

        # Load data based on format
        logger.info(f"[{analysis_id}] Loading data file ({len(data_content)} bytes) as {data_format}")

        try:
            if data_format == DataFormatEnum.csv:
                df = mcp_server.DataLoader.load_csv(data_content)
            elif data_format == DataFormatEnum.json:
                df = mcp_server.DataLoader.load_json(data_content)
            elif data_format == DataFormatEnum.excel:
                # For Excel, we need to save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                    tmp.write(data_content)
                    tmp_path = tmp.name
                try:
                    df = pd.read_excel(tmp_path)
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
            elif data_format == DataFormatEnum.parquet:
                # For Parquet, we need to save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as tmp:
                    tmp.write(data_content)
                    tmp_path = tmp.name
                try:
                    df = pd.read_parquet(tmp_path)
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported data format: {data_format}"
                )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to load data file: {str(e)}"
            )
        except Exception as e:
            logger.error(f"[{analysis_id}] Error loading data: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing data file: {str(e)}"
            )

        logger.info(f"[{analysis_id}] Data loaded: {len(df)} rows, {len(df.columns)} columns")

        # -------------------------------------------------------------------------
        # Step 2: Parse dictionary if provided
        # -------------------------------------------------------------------------

        schema = None
        rules = None
        program = None

        if dictionary_file:
            logger.info(f"[{analysis_id}] Dictionary file provided: {dictionary_file.filename}")

            # Validate dictionary file size (10 MB max)
            dict_content = await dictionary_file.read()
            if len(dict_content) > 10 * 1024 * 1024:  # 10 MB
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Dictionary file too large. Maximum size is 10 MB."
                )

            # Check if we have program manager for dictionary parsing
            if program_manager:
                try:
                    # Parse dictionary and create/find program
                    program = program_manager.create_program_from_dictionary(
                        dictionary_content=dict_content.decode('utf-8'),
                        dictionary_path=Path(dictionary_file.filename),
                        save=True
                    )
                    logger.info(f"[{analysis_id}] Program created/found: {program.name}")

                    # Extract schema and rules from program
                    schema = program.schema
                    rules = program.conditional_rules
                except Exception as e:
                    logger.warning(f"[{analysis_id}] Failed to parse dictionary: {e}")
                    # Continue without dictionary - will run basic analysis only
            else:
                logger.warning(f"[{analysis_id}] Program manager not available, skipping dictionary parsing")

        # -------------------------------------------------------------------------
        # Step 3: Run basic quality checks
        # -------------------------------------------------------------------------

        logger.info(f"[{analysis_id}] Running QualityPipeline checks")
        pipeline = mcp_server.QualityPipeline(df, schema=schema, rules=rules)
        quality_results = pipeline.run_all_checks(min_rows=1)

        # -------------------------------------------------------------------------
        # Step 4: Run logic validation if requested and program available
        # -------------------------------------------------------------------------

        logic_violations = []

        if validate_logic and program and logic_validator:
            logger.info(f"[{analysis_id}] Running logic validation")
            try:
                # Run logic validation using the program
                validation_results = logic_validator.validate_data(df, program.generated_code)

                # Convert logic violations to API model format
                for violation in validation_results.get('violations', []):
                    logic_violations.append(LogicViolation(
                        rule_id=violation.get('rule_id', 'unknown'),
                        rule_description=violation.get('rule_description', ''),
                        row_index=violation.get('row_index', 0),
                        affected_fields=violation.get('affected_fields', []),
                        actual_values=violation.get('actual_values', {}),
                        expected_behavior=violation.get('expected_behavior', ''),
                        severity=SeverityEnum(violation.get('severity', 'error'))
                    ))

                logger.info(f"[{analysis_id}] Logic validation found {len(logic_violations)} violations")
            except Exception as e:
                logger.error(f"[{analysis_id}] Logic validation failed: {e}")
                # Continue with basic analysis results

        # -------------------------------------------------------------------------
        # Step 5: Build comprehensive response
        # -------------------------------------------------------------------------

        # Convert quality check issues to field violations
        field_violations = []
        for issue in quality_results.get('issues', []):
            field_violations.append(FieldViolation(
                field_name=issue.get('column', 'unknown'),
                row_index=issue.get('violating_rows', [0])[0] if issue.get('violating_rows') else 0,
                violation_type=issue.get('issue', 'unknown'),
                expected=str(issue.get('expected_type', issue.get('rule', ''))),
                actual=issue.get('actual_type', issue.get('sample_values', '')),
                severity=SeverityEnum.error
            ))

        # Generate recommendations based on issues found
        recommendations = []
        if not quality_results.get('overall_passed', False):
            if any(issue.get('issue') == 'type_mismatch' for issue in quality_results.get('issues', [])):
                recommendations.append("Review data types - some columns have type mismatches")
            if any(issue.get('rule', '').startswith('min') or issue.get('rule', '').startswith('max')
                   for issue in quality_results.get('issues', [])):
                recommendations.append("Check value ranges - some values are out of expected bounds")

        if len(logic_violations) > 0:
            recommendations.append(f"Review {len(logic_violations)} conditional logic violations")

        if len(df) == 0:
            recommendations.append("Dataset is empty - verify data file content")

        # Calculate execution time
        execution_time = time.time() - start_time

        # Build response
        response = AnalyzeResponse(
            analysis_id=analysis_id,
            timestamp=datetime.now(),
            summary=AnalysisSummary(
                total_rows=len(df),
                total_columns=len(df.columns),
                issues_found=len(field_violations),
                logic_violations=len(logic_violations),
                execution_time_seconds=round(execution_time, 2)
            ),
            field_violations=field_violations,
            logic_violations=logic_violations,
            recommendations=recommendations,
            program_used=program.name if program else None
        )

        logger.info(f"[{analysis_id}] Analysis complete: {len(field_violations)} field issues, "
                   f"{len(logic_violations)} logic violations, {execution_time:.2f}s")

        return response

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log and return internal server error for unexpected exceptions
        logger.error(f"[{analysis_id}] Unexpected error: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/api/v1/analyze/with-program", response_model=AnalyzeResponse)
@limiter.limit("10/minute")
async def analyze_with_program(
    request: Request,
    data_file: UploadFile = File(..., description="Data file to analyze"),
    program: str = Form(..., description="Program name, ID, or alias"),
    data_format: DataFormatEnum = Form(DataFormatEnum.csv, description="Data file format"),
    return_format: ReturnFormatEnum = Form(ReturnFormatEnum.json, description="Format for results"),
    api_key: str = Depends(verify_api_key)
):
    """
    Analyze data using a cached validation program

    **Authentication**: Requires API key via X-API-Key header

    **Rate Limit**: 10 requests per minute

    **Workflow**:
    1. Load validation program by name, ID, or alias
    2. Load and validate data file
    3. Run QualityPipeline with program's schema and rules
    4. Run logic validation using program's generated code
    5. Return comprehensive analysis results

    **Program Lookup**:
    - By name: e.g., "20241202-143022-ClinicalTrial"
    - By ID: e.g., "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    - By alias: e.g., "johnDoesFav01"

    **Deleted Programs**: Returns 404 if program is deleted (soft delete)

    **File Size Limit**: 50 MB max

    **Response**: AnalyzeResponse with field violations, logic violations, and recommendations
    """
    start_time = time.time()
    analysis_id = str(uuid_lib.uuid4())

    logger.info(f"[{analysis_id}] Starting analysis with program: {program}")

    # Validate services are available
    if not program_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Program management service unavailable"
        )

    if not mcp_server:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data analysis service unavailable"
        )

    try:
        # -------------------------------------------------------------------------
        # Step 1: Load validation program
        # -------------------------------------------------------------------------

        logger.info(f"[{analysis_id}] Loading program: {program}")

        try:
            validation_program = program_manager.find_program(program)
            if not validation_program:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Program not found: {program}"
                )

            # Check if program is deleted
            if validation_program.status == 'deleted':
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Program '{program}' has been deleted"
                )

            logger.info(f"[{analysis_id}] Program loaded: {validation_program.name} "
                       f"({validation_program.num_fields} fields, "
                       f"{validation_program.num_logic_rules} logic rules)")

            # Update program usage tracking
            program_manager.db.update_program_usage(validation_program.program_id)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[{analysis_id}] Error loading program: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load program: {str(e)}"
            )

        # -------------------------------------------------------------------------
        # Step 2: Load and validate data file
        # -------------------------------------------------------------------------

        # Check file size (50 MB max)
        data_content = await data_file.read()
        if len(data_content) > 50 * 1024 * 1024:  # 50 MB
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Data file too large. Maximum size is 50 MB."
            )

        logger.info(f"[{analysis_id}] Loading data file ({len(data_content)} bytes) as {data_format}")

        try:
            if data_format == DataFormatEnum.csv:
                df = mcp_server.DataLoader.load_csv(data_content)
            elif data_format == DataFormatEnum.json:
                df = mcp_server.DataLoader.load_json(data_content)
            elif data_format == DataFormatEnum.excel:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                    tmp.write(data_content)
                    tmp_path = tmp.name
                try:
                    df = pd.read_excel(tmp_path)
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
            elif data_format == DataFormatEnum.parquet:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as tmp:
                    tmp.write(data_content)
                    tmp_path = tmp.name
                try:
                    df = pd.read_parquet(tmp_path)
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported data format: {data_format}"
                )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to load data file: {str(e)}"
            )
        except Exception as e:
            logger.error(f"[{analysis_id}] Error loading data: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing data file: {str(e)}"
            )

        logger.info(f"[{analysis_id}] Data loaded: {len(df)} rows, {len(df.columns)} columns")

        # -------------------------------------------------------------------------
        # Step 3: Run quality checks with program's schema and rules
        # -------------------------------------------------------------------------

        logger.info(f"[{analysis_id}] Running QualityPipeline with program schema/rules")
        pipeline = mcp_server.QualityPipeline(
            df,
            schema=validation_program.schema,
            rules=validation_program.conditional_rules
        )
        quality_results = pipeline.run_all_checks(min_rows=1)

        # -------------------------------------------------------------------------
        # Step 4: Run logic validation using program's generated code
        # -------------------------------------------------------------------------

        logic_violations = []

        if validation_program.generated_code and logic_validator:
            logger.info(f"[{analysis_id}] Running logic validation with program code")
            try:
                validation_results = logic_validator.validate_data(
                    df,
                    validation_program.generated_code
                )

                # Convert logic violations to API model format
                for violation in validation_results.get('violations', []):
                    logic_violations.append(LogicViolation(
                        rule_id=violation.get('rule_id', 'unknown'),
                        rule_description=violation.get('rule_description', ''),
                        row_index=violation.get('row_index', 0),
                        affected_fields=violation.get('affected_fields', []),
                        actual_values=violation.get('actual_values', {}),
                        expected_behavior=violation.get('expected_behavior', ''),
                        severity=SeverityEnum(violation.get('severity', 'error'))
                    ))

                logger.info(f"[{analysis_id}] Logic validation found {len(logic_violations)} violations")
            except Exception as e:
                logger.error(f"[{analysis_id}] Logic validation failed: {e}")
                # Continue with basic analysis results

        # -------------------------------------------------------------------------
        # Step 5: Build comprehensive response
        # -------------------------------------------------------------------------

        # Convert quality check issues to field violations
        field_violations = []
        for issue in quality_results.get('issues', []):
            field_violations.append(FieldViolation(
                field_name=issue.get('column', 'unknown'),
                row_index=issue.get('violating_rows', [0])[0] if issue.get('violating_rows') else 0,
                violation_type=issue.get('issue', 'unknown'),
                expected=str(issue.get('expected_type', issue.get('rule', ''))),
                actual=issue.get('actual_type', issue.get('sample_values', '')),
                severity=SeverityEnum.error
            ))

        # Generate recommendations
        recommendations = []
        if not quality_results.get('overall_passed', False):
            if any(issue.get('issue') == 'type_mismatch' for issue in quality_results.get('issues', [])):
                recommendations.append("Review data types - some columns have type mismatches")
            if any(issue.get('rule', '').startswith('min') or issue.get('rule', '').startswith('max')
                   for issue in quality_results.get('issues', [])):
                recommendations.append("Check value ranges - some values are out of expected bounds")

        if len(logic_violations) > 0:
            recommendations.append(f"Review {len(logic_violations)} conditional logic violations")

        if len(df) == 0:
            recommendations.append("Dataset is empty - verify data file content")

        # Calculate execution time
        execution_time = time.time() - start_time

        # Build response
        response = AnalyzeResponse(
            analysis_id=analysis_id,
            timestamp=datetime.now(),
            summary=AnalysisSummary(
                total_rows=len(df),
                total_columns=len(df.columns),
                issues_found=len(field_violations),
                logic_violations=len(logic_violations),
                execution_time_seconds=round(execution_time, 2)
            ),
            field_violations=field_violations,
            logic_violations=logic_violations,
            recommendations=recommendations,
            program_used=validation_program.name
        )

        logger.info(f"[{analysis_id}] Analysis complete: {len(field_violations)} field issues, "
                   f"{len(logic_violations)} logic violations, {execution_time:.2f}s")

        return response

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log and return internal server error for unexpected exceptions
        logger.error(f"[{analysis_id}] Unexpected error: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )
