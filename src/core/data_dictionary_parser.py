#!/usr/bin/env python3
"""
Data Dictionary Parser with LLM Code Generation
Generates deterministic Python code to parse data dictionaries
rather than parsing them directly for verifiable, cacheable results
"""

import asyncio
import json
import hashlib
import subprocess
import sys
import tempfile
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
import re

# Azure OpenAI imports
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of dictionary validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    confidence_score: float  # 0.0 to 1.0

class DataDictionaryParser:
    """Parse data dictionaries using LLM-generated code for deterministic results"""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize parser with optional cache directory"""
        self.cache_dir = cache_dir or Path.home() / ".cache" / "data_analyzer" / "parsers"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Azure OpenAI client
        self.client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini")

        # Track parsing attempts for retry logic
        self.max_retries = 3

    async def parse_dictionary(
        self,
        content: str,
        format_hint: str = "auto",
        use_cache: bool = True,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        Parse a data dictionary using LLM-generated code

        Args:
            content: The dictionary content to parse
            format_hint: Hint about format (csv, json, excel, redcap, etc.)
            use_cache: Whether to use cached parsers
            debug: Enable debug output

        Returns:
            Parsed dictionary with schema, rules, and metadata
        """
        # Generate cache key based on content structure
        cache_key = self._generate_cache_key(content, format_hint)

        # Check cache if enabled
        if use_cache:
            cached_parser = self._load_cached_parser(cache_key)
            if cached_parser:
                logger.info(f"Using cached parser for key: {cache_key}")
                return await self._execute_parser(cached_parser, content, debug)

        # Generate new parser code
        parser_code = await self._generate_parser_code(content, format_hint)

        if debug:
            logger.debug(f"Generated parser code:\n{parser_code}")

        # Validate and execute the parser
        result = await self._execute_parser(parser_code, content, debug)

        # Add the generated parser code to the result for download
        if result.get("success"):
            result["parser_code"] = parser_code
            # Cache successful parser
            if use_cache:
                self._cache_parser(cache_key, parser_code)

        return result

    def _generate_cache_key(self, content: str, format_hint: str) -> str:
        """Generate a cache key based on content structure"""
        # Extract structural elements (headers, patterns, etc.)
        lines = content.strip().split('\n')[:10]  # First 10 lines for structure
        structure = f"{format_hint}:{':'.join(lines)}"

        # Create hash
        return hashlib.sha256(structure.encode()).hexdigest()[:16]

    def _load_cached_parser(self, cache_key: str) -> Optional[str]:
        """Load cached parser code if it exists"""
        cache_file = self.cache_dir / f"{cache_key}.py"
        if cache_file.exists():
            # Check if cache is not too old (365 days / 1 year)
            if (datetime.now().timestamp() - cache_file.stat().st_mtime) < 365 * 86400:
                return cache_file.read_text()
        return None

    def _cache_parser(self, cache_key: str, parser_code: str):
        """Cache parser code for future use"""
        cache_file = self.cache_dir / f"{cache_key}.py"
        cache_file.write_text(parser_code)

        # TODO: Implement cache size management
        # - Monitor total cache size
        # - Implement LRU eviction when size exceeds threshold
        # - Add cache metrics/statistics

    async def _generate_parser_code(self, content: str, format_hint: str) -> str:
        """Generate Python code to parse the dictionary using LLM"""

        prompt = self._get_parser_generation_prompt(content, format_hint)

        try:
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert Python developer specializing in data parsing and validation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for more deterministic output
                max_tokens=2000
            )

            # Extract code from response
            code = response.choices[0].message.content

            # Clean up code (remove markdown blocks if present)
            code = self._extract_python_code(code)

            return code

        except Exception as e:
            logger.error(f"Error generating parser code: {str(e)}")
            raise

    def _get_parser_generation_prompt(self, content: str, format_hint: str) -> str:
        """Generate the prompt for LLM code generation"""

        # Truncate content if too long
        sample_content = content[:2000] if len(content) > 2000 else content

        prompt = f"""Generate Python code to parse this data dictionary into a structured format.

Format hint: {format_hint}

Data Dictionary Sample:
{sample_content}

Generate ONLY executable Python code with these requirements:

1. Define a function: def parse_dictionary(content: str) -> dict
2. The function must handle the exact input format shown above
3. Return a dictionary with this EXACT structure:
{{
    "columns": {{
        "column_name": {{
            "type": "int|float|str|bool|datetime|date|decimal|array",
            "required": true|false,
            "description": "column description",
            "min": numeric_min (optional),
            "max": numeric_max (optional),
            "allowed_values": ["value1", "value2"] (optional),
            "pattern": "regex pattern" (optional),
            "format": "date format like YYYY-MM-DD" (optional),
            "unit": "unit of measurement" (optional),
            "precision": number (for decimals, optional),
            "clinical_type": "demographic|lab|vitals|adverse_event|outcome" (optional for clinical data)
        }}
    }},
    "metadata": {{
        "source_format": "{format_hint}",
        "parsed_at": "ISO timestamp",
        "total_columns": number,
        "version": "version if found"
    }}
}}

4. Handle common dictionary formats:
   - CSV with headers like: Column,Type,Required,Min,Max,Description
   - JSON with field definitions
   - Text-based descriptions
   - REDCap data dictionaries
   - Clinical trial CRF specifications

5. Infer data types from various notations:
   - "integer", "int", "number", "numeric" -> "int"
   - "decimal", "float", "double", "real" -> "float"
   - "string", "text", "varchar", "char" -> "str"
   - "boolean", "bool", "yes/no", "true/false" -> "bool"
   - "date", "datetime", "timestamp" -> "datetime" or "date"
   - Lists/arrays indicated by brackets or "multiple choice"

6. Extract constraints:
   - Numeric ranges (min/max)
   - Allowed values / choice lists
   - Required/optional fields
   - Patterns/formats

7. Handle errors gracefully - return partial results if possible

DO NOT include:
- Import statements (assume standard library available)
- Example usage
- Comments or docstrings
- Print statements
- Any text outside the function definition

The code will be executed with: result = parse_dictionary(content)
"""
        return prompt

    def _extract_python_code(self, text: str) -> str:
        """Extract Python code from LLM response"""
        # Remove markdown code blocks
        if "```python" in text:
            match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
            if match:
                return match.group(1)
        elif "```" in text:
            match = re.search(r"```\n(.*?)\n```", text, re.DOTALL)
            if match:
                return match.group(1)

        # Return as-is if no markdown blocks
        return text.strip()

    async def _execute_parser(
        self,
        parser_code: str,
        content: str,
        debug: bool = False
    ) -> Dict[str, Any]:
        """Execute parser code in subprocess sandbox"""

        # TODO: Future security enhancements:
        # - Use Docker container for complete isolation
        # - Implement resource limits (memory, CPU)
        # - Create whitelist of allowed imports
        # - Use ast.parse() to validate code structure before execution
        # - Run in restricted Python environment (RestrictedPython)

        try:
            # Create temporary file with parser code and execution
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Write the parser function
                f.write(parser_code)
                f.write("\n\n")

                # Write the content and execution code
                f.write(f"content = '''{content}'''\n")
                f.write("import json\n")
                f.write("from datetime import datetime\n")
                f.write("try:\n")
                f.write("    result = parse_dictionary(content)\n")
                f.write("    result['success'] = True\n")
                f.write("    print(json.dumps(result, default=str))\n")
                f.write("except Exception as e:\n")
                f.write("    import traceback\n")
                f.write("    error_result = {\n")
                f.write("        'success': False,\n")
                f.write("        'error': str(e),\n")
                f.write("        'traceback': traceback.format_exc()\n")
                f.write("    }\n")
                f.write("    print(json.dumps(error_result))\n")

                temp_file = f.name

            # Execute in subprocess with timeout
            # 600 second (10 minute) timeout for complex parsing operations
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=600  # 600 second (10 minute) timeout
            )

            # Clean up temp file
            os.unlink(temp_file)

            if result.returncode == 0:
                parsed = json.loads(result.stdout)

                if parsed.get("success"):
                    # Validate the parsed output
                    validation = self._validate_parsed_output(parsed)
                    parsed["validation"] = {
                        "is_valid": validation.is_valid,
                        "errors": validation.errors,
                        "warnings": validation.warnings,
                        "confidence_score": validation.confidence_score
                    }
                    # Also add confidence score at top level for easier access
                    parsed["confidence_score"] = validation.confidence_score

                return parsed
            else:
                error_msg = result.stderr or "Unknown execution error"
                if debug:
                    logger.error(f"Parser execution failed: {error_msg}")

                return {
                    "success": False,
                    "error": error_msg,
                    "validation": {
                        "is_valid": False,
                        "errors": [error_msg],
                        "warnings": [],
                        "confidence_score": 0.0
                    }
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Parser execution timeout (600 seconds)",
                "validation": {
                    "is_valid": False,
                    "errors": ["Execution timeout"],
                    "warnings": [],
                    "confidence_score": 0.0
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "validation": {
                    "is_valid": False,
                    "errors": [str(e)],
                    "warnings": [],
                    "confidence_score": 0.0
                }
            }

    def _validate_parsed_output(self, parsed: Dict[str, Any]) -> ValidationResult:
        """Validate the structure and content of parsed dictionary"""
        errors = []
        warnings = []
        confidence_score = 1.0

        # Check required structure
        if "columns" not in parsed:
            errors.append("Missing required 'columns' key")
            confidence_score -= 0.5
        else:
            columns = parsed["columns"]

            if not isinstance(columns, dict):
                errors.append("'columns' must be a dictionary")
                confidence_score -= 0.3
            elif len(columns) == 0:
                warnings.append("No columns found in dictionary")
                confidence_score -= 0.2
            else:
                # Validate each column definition
                for col_name, col_def in columns.items():
                    if not isinstance(col_def, dict):
                        errors.append(f"Column {col_name}: definition must be a dictionary")
                        confidence_score -= 0.1
                        continue

                    # Check for required type field
                    if "type" not in col_def:
                        warnings.append(f"Column {col_name}: missing 'type' field")
                        confidence_score -= 0.05
                    else:
                        valid_types = ["int", "float", "str", "bool", "datetime", "date", "decimal", "array"]
                        if col_def["type"] not in valid_types:
                            warnings.append(f"Column {col_name}: unknown type '{col_def['type']}'")
                            confidence_score -= 0.02

                    # Validate numeric constraints
                    if col_def.get("type") in ["int", "float", "decimal"]:
                        if "allowed_values" in col_def and ("min" in col_def or "max" in col_def):
                            warnings.append(
                                f"Column {col_name}: has both range constraints and allowed_values"
                            )
                            confidence_score -= 0.01

                        # Check min/max validity
                        if "min" in col_def and "max" in col_def:
                            try:
                                if float(col_def["min"]) > float(col_def["max"]):
                                    errors.append(f"Column {col_name}: min > max")
                                    confidence_score -= 0.05
                            except (ValueError, TypeError):
                                warnings.append(f"Column {col_name}: invalid min/max values")
                                confidence_score -= 0.02

                    # Validate boolean required field
                    if "required" in col_def and not isinstance(col_def["required"], bool):
                        warnings.append(f"Column {col_name}: 'required' should be boolean")
                        confidence_score -= 0.01

                    # Check pattern validity for string types
                    if "pattern" in col_def and col_def.get("type") == "str":
                        try:
                            re.compile(col_def["pattern"])
                        except re.error:
                            errors.append(f"Column {col_name}: invalid regex pattern")
                            confidence_score -= 0.05

        # Check metadata (optional but good to have)
        if "metadata" not in parsed:
            warnings.append("Missing metadata section")
            confidence_score -= 0.05

        # Ensure confidence score stays in valid range
        confidence_score = max(0.0, min(1.0, confidence_score))

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence_score=confidence_score
        )

    async def retry_with_feedback(
        self,
        content: str,
        format_hint: str,
        previous_errors: List[str],
        attempt: int = 1
    ) -> Dict[str, Any]:
        """Retry parsing with feedback from previous errors"""

        if attempt > self.max_retries:
            return {
                "success": False,
                "error": f"Failed after {self.max_retries} attempts",
                "previous_errors": previous_errors,
                "validation": {
                    "is_valid": False,
                    "errors": previous_errors,
                    "warnings": [],
                    "confidence_score": 0.0
                }
            }

        # Generate improved prompt with error feedback
        error_feedback = "\n".join(previous_errors)
        improved_prompt = f"""
Previous parsing attempt failed with these errors:
{error_feedback}

Please generate corrected Python code that addresses these issues.

{self._get_parser_generation_prompt(content, format_hint)}
"""

        try:
            # Generate new parser code with feedback
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert Python developer. Fix the parsing errors."},
                    {"role": "user", "content": improved_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )

            parser_code = self._extract_python_code(response.choices[0].message.content)
            result = await self._execute_parser(parser_code, content)

            if result.get("success"):
                return result
            else:
                # Retry with accumulated errors
                all_errors = previous_errors + [result.get("error", "Unknown error")]
                return await self.retry_with_feedback(content, format_hint, all_errors, attempt + 1)

        except Exception as e:
            logger.error(f"Error in retry attempt {attempt}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "validation": {
                    "is_valid": False,
                    "errors": [str(e)],
                    "warnings": [],
                    "confidence_score": 0.0
                }
            }

    def convert_to_schema_and_rules(self, parsed_dict: Dict[str, Any]) -> Tuple[Dict, Dict]:
        """Convert parsed dictionary to schema and rules format"""

        if not parsed_dict.get("success") or "columns" not in parsed_dict:
            return {}, {}

        schema = {}
        rules = {}

        for col_name, col_def in parsed_dict["columns"].items():
            # Extract schema (data type)
            data_type = col_def.get("type", "str")

            # Map extended types to basic types for schema
            type_mapping = {
                "decimal": "float",
                "array": "str",  # Arrays stored as delimited strings
                "date": "datetime"
            }
            schema[col_name] = type_mapping.get(data_type, data_type)

            # Extract validation rules
            col_rules = {}

            # Numeric constraints
            if "min" in col_def:
                col_rules["min"] = col_def["min"]
            if "max" in col_def:
                col_rules["max"] = col_def["max"]

            # Categorical constraints
            if "allowed_values" in col_def:
                col_rules["allowed"] = col_def["allowed_values"]

            # Pattern constraints
            if "pattern" in col_def:
                col_rules["pattern"] = col_def["pattern"]

            # Required field
            if col_def.get("required", False):
                col_rules["required"] = True

            # Format hints
            if "format" in col_def:
                col_rules["format"] = col_def["format"]

            # Add rules if any constraints exist
            if col_rules:
                rules[col_name] = col_rules

        return schema, rules


# Async wrapper for use in non-async contexts
def parse_dictionary_sync(
    content: str,
    format_hint: str = "auto",
    use_cache: bool = True,
    debug: bool = False
) -> Dict[str, Any]:
    """Synchronous wrapper for parse_dictionary"""
    parser = DataDictionaryParser()
    return asyncio.run(parser.parse_dictionary(content, format_hint, use_cache, debug))


if __name__ == "__main__":
    # Example usage for testing
    test_dict = """Column,Type,Required,Min,Max,Description
employee_id,integer,Yes,1,999999,Unique employee identifier
first_name,string,Yes,,,Employee first name
last_name,string,Yes,,,Employee last name
salary,decimal,Yes,30000,500000,Annual salary in USD
department,string,Yes,,,Must be: HR|Engineering|Sales|Marketing
hire_date,date,Yes,,,Date of hire (YYYY-MM-DD)
is_active,boolean,Yes,,,Current employment status"""

    # Test the parser
    result = parse_dictionary_sync(test_dict, "csv", use_cache=False, debug=True)
    print(json.dumps(result, indent=2, default=str))