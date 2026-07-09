"""
Program Manager - High-level interface for program management

Orchestrates:
- LLMDictionaryParser (parse dictionary)
- ProgramDatabase (save/load programs)
- Auto-naming and description generation
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib
import getpass
import uuid
import re
import time
import logging

from src.program_cache import ProgramDatabase, ValidationProgram
from src.llm_client import LLMDictionaryParser

logger = logging.getLogger(__name__)


class ProgramManager:
    """High-level interface for program management.

    Orchestrates the complete workflow from dictionary parsing to program
    execution, including automatic naming, caching, and usage tracking.

    Key responsibilities:
    - Parse data dictionaries using LLM
    - Generate meaningful program names and descriptions
    - Manage program lifecycle (create, find, execute, delete)
    - Track program usage and execution history
    - Handle graceful degradation for missing/deleted programs
    """

    def __init__(self, llm_client: Optional[LLMDictionaryParser] = None):
        """Initialize manager with optional LLM client.

        Args:
            llm_client: LLM parser for dictionaries. If None, creates new instance.
                       Will fail gracefully if Azure credentials are not configured.
        """
        try:
            self.llm = llm_client or LLMDictionaryParser()
            logger.info("ProgramManager initialized with LLM client")
        except ValueError as e:
            logger.warning(f"Failed to initialize LLM client: {e}")
            logger.warning("ProgramManager will operate in limited mode")
            self.llm = None
        except Exception as e:
            logger.error(f"Unexpected error initializing LLM client: {e}")
            self.llm = None

        self.db = ProgramDatabase()
        logger.info("ProgramManager initialized with database")

    def create_program_from_dictionary(
        self,
        dictionary_content: str,
        dictionary_path: Optional[Path] = None,
        save: bool = True
    ) -> ValidationProgram:
        """
        Full pipeline: Parse dictionary → Extract rules → Generate code → Save program

        This method orchestrates the complete workflow:
        1. Parse dictionary using LLM to extract field definitions and rules
        2. Calculate dictionary hash for caching
        3. Generate meaningful name and description
        4. Create ValidationProgram object with metadata
        5. Save to database (optional)

        Args:
            dictionary_content: Raw text content of the dictionary
            dictionary_path: Optional path to dictionary file (for naming)
            save: Whether to save program to database

        Returns:
            ValidationProgram with auto-generated name and metadata

        Raises:
            RuntimeError: If LLM client is not available
            Exception: If dictionary parsing fails
        """
        if self.llm is None:
            raise RuntimeError(
                "LLM client not available. Cannot parse dictionary. "
                "Please configure Azure OpenAI credentials."
            )

        start_time = time.time()
        logger.info(f"Starting program creation from dictionary "
                   f"({len(dictionary_content)} characters)")

        # 1. Parse dictionary with LLM (existing functionality)
        try:
            parsed = self.llm.parse_dictionary(dictionary_content)
            logger.info(f"Parsed dictionary: {parsed.get('metadata', {}).get('total_fields', 0)} fields")
        except Exception as e:
            logger.error(f"Failed to parse dictionary: {e}")
            raise RuntimeError(f"Dictionary parsing failed: {e}") from e

        # 2. Calculate dictionary hash for caching/deduplication
        dict_hash = hashlib.md5(dictionary_content.encode()).hexdigest()
        logger.debug(f"Dictionary hash: {dict_hash}")

        # 3. Generate meaningful name
        name = self._generate_name(parsed, dictionary_path)
        logger.info(f"Generated program name: {name}")

        # 4. Build ValidationProgram object
        program = ValidationProgram(
            program_id=self._generate_id(),
            name=name,
            aliases=[],
            dictionary_source=dictionary_path.name if dictionary_path else "uploaded",
            dictionary_hash=dict_hash,
            dictionary_format=self._detect_format(dictionary_content),
            generated_code="",  # Feature 2 will implement code generation
            schema=parsed.get('schema', {}),
            conditional_rules=[],  # Feature 2 will extract conditional logic
            created_by=getpass.getuser(),
            created_at=datetime.now(),
            model_used=self.llm.deployment if hasattr(self.llm, 'deployment') else 'unknown',
            generation_time_seconds=time.time() - start_time,
            num_fields=len(parsed.get('fields', [])),
            num_basic_rules=self._count_basic_rules(parsed),
            num_logic_rules=0  # Feature 2 will populate from conditional_rules
        )

        # 5. Save to database
        if save:
            try:
                self.db.save_program(program)
                logger.info(f"Created and saved program: {program.name} (ID: {program.program_id})")
            except Exception as e:
                logger.error(f"Failed to save program: {e}")
                raise
        else:
            logger.info(f"Created program (not saved): {program.name}")

        return program

    def find_or_create_program(
        self,
        dictionary_content: str,
        dictionary_path: Optional[Path] = None
    ) -> ValidationProgram:
        """
        Check if program exists for this dictionary (by hash).
        If yes: return existing program and increment use count.
        If no: create new program.

        This method implements smart caching based on dictionary content hash:
        - Identical dictionaries reuse the same program
        - Programs track usage frequency via use_count
        - Last_used timestamp is updated on each access

        Args:
            dictionary_content: Raw dictionary content
            dictionary_path: Optional path for naming new programs

        Returns:
            Existing or newly created ValidationProgram

        Raises:
            RuntimeError: If LLM client not available (for new programs)
            Exception: If program creation fails
        """
        # Calculate hash for lookup
        dict_hash = hashlib.md5(dictionary_content.encode()).hexdigest()
        logger.debug(f"Searching for existing program with hash: {dict_hash}")

        # Search by hash
        try:
            existing = self.db.search_programs(dictionary_hash=dict_hash)
        except Exception as e:
            logger.error(f"Error searching for existing programs: {e}")
            # Continue to create new program rather than failing
            existing = []

        if existing:
            # Return first match (there should only be one active per hash)
            program = existing[0]

            # Update last_used and increment use count
            try:
                self.db.increment_use_count(program.program_id)
                logger.info(f"Found existing program: {program.name} "
                           f"(use_count: {program.use_count + 1})")
            except Exception as e:
                logger.warning(f"Failed to update program use count: {e}")

            return program
        else:
            # Create new program
            logger.info("No existing program found, creating new one")
            return self.create_program_from_dictionary(
                dictionary_content,
                dictionary_path,
                save=True
            )

    def execute_program(
        self,
        program_id_or_alias: str,
        data: 'pd.DataFrame'
    ) -> Dict[str, Any]:
        """
        Load program and execute validation on data.

        Handles graceful degradation:
        - If program deleted → return error with suggestion to recreate
        - If program not found → return error with search hints
        - If execution fails → return error with details

        Args:
            program_id_or_alias: Program identifier (ID, name, or alias)
            data: DataFrame to validate

        Returns:
            Dict with keys:
            - 'program': ValidationProgram object
            - 'logic_violations': List of violation records
            - 'summary': Execution summary statistics
            OR
            - 'error': Error type code
            - 'message': Human-readable error message
            - Additional context fields depending on error type
        """
        import pandas as pd  # Import here to avoid circular imports

        start_time = time.time()
        logger.info(f"Executing program: {program_id_or_alias}")

        # Load program from database
        try:
            program = self.db.load_program(program_id_or_alias)
        except Exception as e:
            logger.error(f"Error loading program: {e}")
            return {
                'error': 'database_error',
                'message': f'Failed to load program: {e}'
            }

        # Check if program exists
        if not program:
            logger.warning(f"Program not found: {program_id_or_alias}")
            return {
                'error': 'program_not_found',
                'message': f'Program "{program_id_or_alias}" not found',
                'suggestion': 'Use search_programs() to find available programs, '
                             'or upload a dictionary to create a new program'
            }

        # Check if program was deleted
        if program.status == "deleted":
            logger.warning(f"Attempted to execute deleted program: {program.name}")
            return {
                'error': 'program_deleted',
                'message': f'Program "{program.name}" has been deleted',
                'deleted_at': program.deleted_at.isoformat() if program.deleted_at else None,
                'deletion_reason': program.deletion_reason,
                'suggestion': 'Upload the dictionary again to create a new program'
            }

        # Validate data input
        if not isinstance(data, pd.DataFrame):
            logger.error(f"Invalid data type: {type(data)}")
            return {
                'error': 'invalid_data',
                'message': f'Expected pandas DataFrame, got {type(data).__name__}'
            }

        if data.empty:
            logger.warning("Empty DataFrame provided for validation")
            return {
                'error': 'empty_data',
                'message': 'Cannot validate empty DataFrame'
            }

        # Execute validation logic
        # NOTE: Feature 2 will implement actual validation code execution
        # For now, return empty violations
        try:
            logic_violations = []
            logger.info(f"Validation complete: {len(logic_violations)} violations found")
        except Exception as e:
            logger.error(f"Validation execution failed: {e}")
            return {
                'error': 'execution_failed',
                'message': f'Program execution failed: {e}',
                'program_name': program.name
            }

        # Record execution metadata
        execution_time = time.time() - start_time
        try:
            self.db.increment_use_count(program.program_id)
            self.db.record_execution(program.program_id, {
                'rows_processed': len(data),
                'logic_violations_found': len(logic_violations),
                'executed_by': getpass.getuser(),
                'execution_time_seconds': execution_time
            })
            logger.info(f"Recorded execution: {len(data)} rows, "
                       f"{execution_time:.2f}s")
        except Exception as e:
            logger.warning(f"Failed to record execution metadata: {e}")

        # Return results
        return {
            'program': program,
            'logic_violations': logic_violations,
            'summary': {
                'total_logic_violations': len(logic_violations),
                'program_name': program.name,
                'program_id': program.program_id,
                'rows_processed': len(data),
                'execution_time_seconds': execution_time,
                'executed_at': datetime.now().isoformat()
            }
        }

    def search_programs(
        self,
        query: Optional[str] = None,
        dictionary_source: Optional[str] = None,
        status: str = "active"
    ) -> List[ValidationProgram]:
        """
        Search programs with filters.

        Args:
            query: Text search in program name or description
            dictionary_source: Filter by source filename
            status: Filter by status ("active" or "deleted")

        Returns:
            List of matching ValidationProgram objects
        """
        try:
            results = self.db.search_programs(
                query=query,
                dictionary_source=dictionary_source,
                status=status
            )
            logger.info(f"Search returned {len(results)} programs")
            return results
        except Exception as e:
            logger.error(f"Error searching programs: {e}")
            return []

    def create_alias(self, program_id_or_name: str, alias: str) -> bool:
        """
        Create alias for program.

        Args:
            program_id_or_name: Program identifier to alias
            alias: New alias name

        Returns:
            True if alias created successfully, False if alias exists or program not found
        """
        # First resolve to program_id if name was given
        try:
            program = self.db.load_program(program_id_or_name)
        except Exception as e:
            logger.error(f"Error loading program: {e}")
            return False

        if not program:
            logger.warning(f"Cannot create alias: program '{program_id_or_name}' not found")
            return False

        try:
            result = self.db.create_alias(
                program.program_id,
                alias,
                getpass.getuser()
            )
            if result:
                logger.info(f"Created alias '{alias}' for program {program.name}")
            else:
                logger.warning(f"Alias '{alias}' already exists")
            return result
        except Exception as e:
            logger.error(f"Error creating alias: {e}")
            return False

    def delete_program(
        self,
        program_id_or_name: str,
        reason: str,
        admin_password: str
    ) -> bool:
        """
        Delete program (admin only).

        Args:
            program_id_or_name: Program identifier
            reason: Reason for deletion (audit trail)
            admin_password: Admin password for authorization

        Returns:
            True if deleted successfully, False otherwise
        """
        # First resolve to program_id if name was given
        try:
            program = self.db.load_program(program_id_or_name)
        except Exception as e:
            logger.error(f"Error loading program: {e}")
            return False

        if not program:
            logger.warning(f"Cannot delete: program '{program_id_or_name}' not found")
            return False

        try:
            result = self.db.delete_program(
                program.program_id,
                getpass.getuser(),
                reason,
                admin_password
            )
            if result:
                logger.info(f"Deleted program {program.name}: {reason}")
            else:
                logger.warning("Program deletion failed (incorrect password or other error)")
            return result
        except Exception as e:
            logger.error(f"Error deleting program: {e}")
            return False

    # === Private helper methods ===

    def _generate_name(
        self,
        parsed_dict: Dict,
        dictionary_path: Optional[Path] = None
    ) -> str:
        """
        Generate meaningful name from dictionary content.

        Format: YYYYMMDD-HHMMSS-{description}

        Description generation strategies (in order of preference):
        1. Use dictionary name from metadata
        2. Use source filename (without extension)
        3. Analyze field names for domain (clinical, employee, etc.)
        4. Default to "DataValidation"

        Args:
            parsed_dict: Parsed dictionary from LLM
            dictionary_path: Optional source file path

        Returns:
            Name like "20241202-143022-ClinicalTrial" or "20241202-151530-EmployeeRecords"
        """
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        description = self._generate_description(parsed_dict, dictionary_path)
        full_name = f"{timestamp}-{description}"

        logger.debug(f"Generated program name: {full_name}")
        return full_name

    def _generate_description(
        self,
        parsed_dict: Dict,
        dictionary_path: Optional[Path] = None
    ) -> str:
        """
        Generate meaningful description from dictionary content.

        Uses multiple strategies to infer domain/purpose from dictionary metadata
        and field names.

        Args:
            parsed_dict: Parsed dictionary from LLM
            dictionary_path: Optional source file path

        Returns:
            Clean description string suitable for program name
        """
        # Strategy 1: Use dictionary name from metadata
        if 'metadata' in parsed_dict and 'dictionary_name' in parsed_dict['metadata']:
            name = parsed_dict['metadata']['dictionary_name']
            if name and isinstance(name, str):
                clean = self._clean_name(name)
                if clean and clean != "DataValidation":
                    logger.debug(f"Using metadata name: {clean}")
                    return clean

        # Strategy 2: Use source filename
        if dictionary_path:
            stem = dictionary_path.stem
            if stem and len(stem) > 3:
                clean = self._clean_name(stem)
                if clean and clean != "DataValidation":
                    logger.debug(f"Using filename: {clean}")
                    return clean

        # Strategy 3: Analyze field names for domain keywords
        fields = [f.get('field_name', '') for f in parsed_dict.get('fields', [])]
        fields_text = ' '.join(fields).lower()

        # Clinical/Medical domain keywords
        clinical_keywords = [
            'patient', 'diagnosis', 'treatment', 'adverse', 'medical',
            'clinical', 'therapy', 'symptom', 'drug', 'dose', 'pregnant',
            'subject', 'visit', 'baseline', 'followup', 'protocol',
            'trial', 'study', 'arm', 'randomization'
        ]
        if any(kw in fields_text for kw in clinical_keywords):
            logger.debug("Detected clinical domain from field names")
            return "ClinicalData"

        # Employee/HR domain keywords
        employee_keywords = [
            'employee', 'salary', 'hire', 'department', 'manager',
            'position', 'job', 'payroll', 'hr', 'staff', 'personnel',
            'benefits', 'performance', 'review'
        ]
        if any(kw in fields_text for kw in employee_keywords):
            logger.debug("Detected employee/HR domain from field names")
            return "EmployeeData"

        # Survey/Research domain keywords
        survey_keywords = [
            'survey', 'question', 'response', 'score', 'rating',
            'feedback', 'opinion', 'questionnaire', 'respondent',
            'likert', 'scale'
        ]
        if any(kw in fields_text for kw in survey_keywords):
            logger.debug("Detected survey domain from field names")
            return "SurveyData"

        # Financial domain keywords
        financial_keywords = [
            'account', 'transaction', 'balance', 'credit',
            'debit', 'invoice', 'payment', 'revenue', 'expense',
            'ledger', 'fiscal', 'budget'
        ]
        if any(kw in fields_text for kw in financial_keywords):
            logger.debug("Detected financial domain from field names")
            return "FinancialData"

        # Laboratory/Scientific domain keywords
        lab_keywords = [
            'sample', 'specimen', 'assay', 'result', 'test',
            'measurement', 'lab', 'laboratory', 'experiment',
            'concentration', 'unit'
        ]
        if any(kw in fields_text for kw in lab_keywords):
            logger.debug("Detected laboratory domain from field names")
            return "LaboratoryData"

        # Default fallback
        logger.debug("No domain detected, using default")
        return "DataValidation"

    def _clean_name(self, name: str) -> str:
        """
        Clean name for use in program identifier.

        - Remove special characters (keep alphanumeric, underscore, hyphen)
        - Limit length to 30 chars
        - Convert to CamelCase if contains spaces/underscores

        Args:
            name: Raw name string

        Returns:
            Cleaned name suitable for program identifier
        """
        if not name or not isinstance(name, str):
            return "DataValidation"

        # Remove special characters, keep alphanumeric, underscore, hyphen, space
        clean = re.sub(r'[^a-zA-Z0-9_\-\s]', '', name)

        # Convert to CamelCase if contains spaces or underscores
        if ' ' in clean or '_' in clean:
            words = re.split(r'[\s_-]+', clean)
            clean = ''.join(word.capitalize() for word in words if word)

        # Limit length
        if len(clean) > 30:
            clean = clean[:30]

        # Ensure not empty
        return clean if clean else "DataValidation"

    def _generate_id(self) -> str:
        """
        Generate unique program ID.

        Returns:
            UUID4 string
        """
        return str(uuid.uuid4())

    def _detect_format(self, content: str) -> str:
        """
        Detect dictionary format from content.

        Recognizes common formats:
        - FHIR JSON (Questionnaire resources)
        - REDCap CSV (data dictionary format)
        - Generic JSON
        - Generic CSV

        Args:
            content: Raw dictionary content

        Returns:
            Format name string
        """
        content_lower = content.lower()
        content_stripped = content.strip()

        # FHIR Questionnaire JSON
        if '"resourcetype"' in content_lower and '"questionnaire"' in content_lower:
            return "FHIR JSON"

        # REDCap CSV format (has specific column headers)
        if 'variable / field name' in content_lower or 'variable_field_name' in content_lower:
            return "REDCap CSV"

        # Generic JSON (starts with { or [)
        if content_stripped.startswith('{') or content_stripped.startswith('['):
            return "JSON"

        # Default to CSV
        return "CSV"

    def _count_basic_rules(self, parsed: Dict) -> int:
        """
        Count field-level validation rules.

        Counts:
        - Min/max value constraints
        - Allowed values (categorical validation)
        - Format patterns
        - Required field rules

        Args:
            parsed: Parsed dictionary from LLM

        Returns:
            Total count of basic validation rules
        """
        count = 0

        for field in parsed.get('fields', []):
            # Min/max value rules
            if field.get('min_value') is not None or field.get('max_value') is not None:
                count += 1

            # Allowed values (categorical)
            if field.get('allowed_values'):
                count += 1

            # Format pattern rules
            if field.get('format_pattern'):
                count += 1

            # Required field rules
            if field.get('required'):
                count += 1

        logger.debug(f"Counted {count} basic validation rules")
        return count
