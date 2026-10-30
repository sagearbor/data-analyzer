"""
Azure OpenAI LLM Client for Data Dictionary Parsing
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import re
import time
import requests  # For Responses API calls
from dotenv import load_dotenv
import tiktoken

# Shared DCRI LLM module (vendored, do not edit): managed-identity bearer tokens
# and one schema-v1 `llm_call` telemetry event per call.
from src.dcri_llm import (
    DEFAULT_MAX_COMPLETION_TOKENS,
    get_bearer_token,
    record_llm_call,
    usage_from_responses_json,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# All Azure OpenAI models now use Responses API (/openai/v1/responses)
# Model-specific output token limits
MODEL_OUTPUT_LIMITS = {
    'gpt-4o-mini': 16384,
    'gpt-4o': 16384,
    'gpt-5': 128000,
    'gpt-5-nano': 128000,
    'gpt-5-mini': 128000,
    'gpt-5-codex': 128000,
}


class LLMCallError(RuntimeError):
    """An LLM call failed, was truncated (finish_reason=length) or returned no text.

    Raised instead of returning an empty result so callers surface the failure
    to the user rather than reporting "0 fields extracted" as success.
    """


def max_output_tokens_for(deployment: str) -> int:
    """Output-token budget for a deployment.

    Known models use their documented ceiling; unknown ones (e.g. gpt-5.4-*)
    get 128000. Reasoning models are never capped below 32000 (HARD RULE:
    thinking tokens come out of this budget).
    """
    limit = MODEL_OUTPUT_LIMITS.get((deployment or "").lower(), 128000)
    if re.match(r"^(gpt-5|o\d)", (deployment or "").lower()):
        limit = max(limit, DEFAULT_MAX_COMPLETION_TOKENS)
    return limit


def get_available_deployments() -> List[str]:
    """
    Parse the list of Azure OpenAI deployments offered to users (e.g. for a
    model-selection dropdown in the UI).

    Reads AZURE_OPENAI_DEPLOYMENTS as a comma-separated list. If unset (or
    empty after stripping), falls back to the single AZURE_OPENAI_DEPLOYMENT
    value so existing single-deployment setups keep working unchanged.

    Returns:
        Ordered list of deployment names with duplicates removed and
        whitespace stripped. Always contains at least one entry.
    """
    default_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano")
    raw = os.getenv("AZURE_OPENAI_DEPLOYMENTS", "")

    names = [d.strip() for d in raw.split(",") if d.strip()]

    if not names:
        return [default_deployment]

    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for name in names:
        if name not in seen:
            seen.add(name)
            deduped.append(name)

    return deduped


@dataclass
class FieldDefinition:
    """Represents a field/column definition extracted from dictionary"""
    field_name: str
    data_type: str  # int, float, str, date, datetime, boolean
    required: bool = False
    description: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: List[str] = None
    format_pattern: Optional[str] = None
    business_rules: List[str] = None
    conditional_rules: List[Dict[str, Any]] = None  # LLM-extracted conditional logic

    def __post_init__(self):
        if self.allowed_values is None:
            self.allowed_values = []
        if self.business_rules is None:
            self.business_rules = []
        if self.conditional_rules is None:
            self.conditional_rules = []

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Remove None values for cleaner output
        return {k: v for k, v in result.items() if v is not None and v != [] and v != ""}


class LLMDictionaryParser:
    """Parse data dictionaries using Azure OpenAI"""

    def __init__(self):
        """Initialize Azure OpenAI Responses API client"""
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano")

        if not self.endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is not set")

        # Note: Using Responses API via requests library (not OpenAI SDK).
        # Auth is a managed-identity bearer token from dcri_llm (no API key).

        # Token counting for gpt-4o-mini
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")

        logger.info(f"LLM client initialized with endpoint: {self.endpoint}")

    def _call_responses_api(self, prompt: str, deployment: str, max_output: int) -> str:
        """POST one request to the Responses API and return the output text.

        Emits exactly one dcri_llm telemetry event (token counts and lengths
        only, never prompt or completion text). Raises LLMCallError on HTTP
        errors, truncation (finish_reason=length) or empty output.
        """
        url = f"{self.endpoint.rstrip('/')}/openai/v1/responses"
        request_body = {
            "model": deployment,
            "input": prompt,  # String, not messages array!
            "max_output_tokens": max_output,  # Not max_tokens!
            "reasoning": {"effort": "low"},  # GPT-5: low/medium/high
            "text": {"verbosity": "medium"},  # minimal/medium/verbose
        }
        base = {
            "api": "responses",
            "endpoint_host": url,
            "model_requested": deployment,
            "max_completion_tokens": max_output,
            "reasoning_effort": "low",
            "input_chars": len(prompt),
            "streamed": False,
        }
        t0 = time.monotonic()
        try:
            response = requests.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + get_bearer_token(),
                },
                json=request_body,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            record_llm_call(
                **base,
                latency_ms=int((time.monotonic() - t0) * 1000),
                outcome="timeout" if isinstance(e, requests.Timeout) else "error",
                error_type=type(e).__name__,
                http_status=status,
            )
            logger.error(f"Responses API call failed ({type(e).__name__}, http_status={status})")
            raise LLMCallError(f"LLM call failed: {type(e).__name__} (HTTP {status})") from e

        usage = usage_from_responses_json(data)
        finish = usage.get("finish_reason")
        failed = finish == "length" or usage.get("empty_output")
        record_llm_call(
            **base,
            latency_ms=int((time.monotonic() - t0) * 1000),
            outcome="error" if failed else "ok",
            error_type=("finish_length" if finish == "length" else "empty_output") if failed else None,
            **usage,
        )
        if finish == "length":
            raise LLMCallError(
                f"LLM output was truncated (finish_reason=length, budget {max_output} tokens, "
                f"{usage.get('reasoning_tokens')} reasoning tokens)"
            )
        if usage.get("empty_output"):
            raise LLMCallError(f"LLM returned no text (status={usage.get('status')})")
        return self.parse_responses_api_output(data)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

    def chunk_text(self, text: str, max_tokens: int = 4500) -> List[str]:
        """
        Split text into chunks that fit within token limits
        Increased to 4500 tokens for better field extraction from large dictionaries
        """
        # Split by common delimiters first
        sections = re.split(r'\n{2,}|\f', text)

        chunks = []
        current_chunk = []
        current_tokens = 0

        for section in sections:
            section_tokens = self.count_tokens(section)

            if section_tokens > max_tokens:
                # Section too large, split by lines
                lines = section.split('\n')
                for line in lines:
                    line_tokens = self.count_tokens(line)
                    if current_tokens + line_tokens > max_tokens:
                        if current_chunk:
                            chunks.append('\n'.join(current_chunk))
                            current_chunk = []
                            current_tokens = 0
                    current_chunk.append(line)
                    current_tokens += line_tokens
            else:
                if current_tokens + section_tokens > max_tokens:
                    if current_chunk:
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = []
                        current_tokens = 0
                current_chunk.append(section)
                current_tokens += section_tokens

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    def create_extraction_prompt(self, text_chunk: str, is_continuation: bool = False) -> str:
        """Create prompt for extracting field definitions from dictionary text"""

        base_prompt = """You are a data analyst expert at extracting structured information from data dictionaries.
Extract field/column definitions from the following text and return them as a JSON object.

For each field, extract:
- field_name: The exact column/field name
- data_type: One of: int, float, str, date, datetime, boolean
- required: true if field is required/mandatory
- description: Field description
- min_value: Minimum value (if specified)
- max_value: Maximum value (if specified)
- allowed_values: List of allowed/valid values (if enumerated). For REDCap "Choices" format like "1, Yes | 0, No", extract ["Yes", "No"] or ["1", "0"] or both.
- format_pattern: Date/time format or regex pattern (if specified)
- business_rules: Any validation rules or business logic
- conditional_rules: Array of conditional logic rules (see below)

CONDITIONAL LOGIC EXTRACTION:
Look for patterns that indicate conditional logic and dependencies between fields:
- "skip if...", "skip when...", "must be blank if...", "leave blank when..."
- "required if...", "required when...", "only if...", "must be filled if..."
- "only shown when...", "only enabled if...", "display if..."
- "allowed if...", "valid only when..."
- Cross-field dependencies (e.g., "if gender=male, skip pregnancy field", "if age >= 18, required")
- Age/date constraints (e.g., "only if age >= 18", "required for subjects over 65")
- Value-based conditions (e.g., "if pregnant=yes, required", "skip if treatment_arm=control")

For each conditional rule, return an object with:
- rule_type: One of "skip_if", "required_if", "show_if", "allowed_if"
- condition_text: The natural language condition (e.g., "gender is male", "age >= 18", "pregnant is yes")
- action: One of "must_be_blank", "must_be_filled", "skip", "required"
- affected_fields: Array of field names affected by this rule

Examples:
- "If male, skip this field" → {"rule_type": "skip_if", "condition_text": "gender is male", "action": "must_be_blank", "affected_fields": ["current_field_name"]}
- "Required for female subjects" → {"rule_type": "required_if", "condition_text": "gender is female", "action": "must_be_filled", "affected_fields": ["current_field_name"]}
- "Only if age >= 18" → {"rule_type": "show_if", "condition_text": "age >= 18", "action": "must_be_filled", "affected_fields": ["current_field_name"]}
- "Skip if treatment arm is control" → {"rule_type": "skip_if", "condition_text": "treatment_arm is control", "action": "must_be_blank", "affected_fields": ["current_field_name"]}

Return a JSON object with a "fields" array containing field definitions. Be precise with field names.
Extract ALL data fields you can identify, even if incomplete.
IMPORTANT for clinical/research dictionaries: Look for fields with patterns like _stop, _date, _dc, _id, etc.
IMPORTANT: For REDCap dictionaries, parse the "Choices, Calculations, OR Slider Labels" column to extract allowed_values.
"""

        continuation_prompt = """Continue extracting field definitions from this dictionary section.
Return only new fields not already processed as a JSON object with "fields" array.
Remember to extract conditional_rules for each field."""

        prompt = continuation_prompt if is_continuation else base_prompt

        return f"""{prompt}

Dictionary text:
```
{text_chunk}
```

Return JSON object with "fields" array:"""

    def repair_json(self, json_text: str) -> str:
        """
        Attempt to repair common JSON errors from truncated/malformed LLM responses.
        Returns repaired JSON string.
        """
        repaired = json_text

        # Fix 1: Add missing closing bracket for arrays
        if repaired.count('[') > repaired.count(']'):
            logger.info("Repairing: Adding missing closing brackets")
            repaired = repaired + ']' * (repaired.count('[') - repaired.count(']'))

        # Fix 2: Add missing closing brace for objects
        if repaired.count('{') > repaired.count('}'):
            logger.info("Repairing: Adding missing closing braces")
            repaired = repaired + '}' * (repaired.count('{') - repaired.count('}'))

        # Fix 3: Remove trailing commas before closing brackets/braces
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)

        # Fix 4: Fix unterminated strings by adding closing quote at end of line
        # This is more aggressive - only use if we're desperate
        lines = repaired.split('\n')
        fixed_lines = []
        for line in lines:
            # Count quotes - if odd, add one at end
            if line.count('"') % 2 == 1:
                line = line.rstrip() + '"'
            fixed_lines.append(line)
        repaired = '\n'.join(fixed_lines)

        return repaired

    def parse_llm_response(self, response_text: str) -> List[FieldDefinition]:
        """Parse LLM response into FieldDefinition objects"""
        fields = []

        try:
            # With json_object mode, response should already be valid JSON
            json_text = response_text.strip()

            # Fallback: Extract JSON from code blocks if wrapped
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)

            # Parse JSON (with repair fallback)
            try:
                parsed = json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error: {e}, attempting repair...")
                try:
                    repaired_text = self.repair_json(json_text)
                    parsed = json.loads(repaired_text)
                    logger.info("✅ Successfully repaired and parsed JSON")
                except Exception as repair_error:
                    logger.error(f"Failed to repair JSON ({len(json_text)} chars): {type(repair_error).__name__}")
                    raise LLMCallError("LLM returned output that is not valid JSON") from repair_error

            # Handle both formats: {"fields": [...]} or [...]
            if isinstance(parsed, dict) and 'fields' in parsed:
                field_data = parsed['fields']
            elif isinstance(parsed, list):
                field_data = parsed
            else:
                logger.error(f"Unexpected JSON format: {type(parsed)}")
                return fields

            # Convert to FieldDefinition objects
            if isinstance(field_data, list):
                for item in field_data:
                    if isinstance(item, dict) and 'field_name' in item:
                        # Map data types
                        dtype = item.get('data_type', 'str').lower()
                        if 'int' in dtype:
                            dtype = 'int'
                        elif 'float' in dtype or 'decimal' in dtype or 'numeric' in dtype:
                            dtype = 'float'
                        elif 'date' in dtype and 'time' in dtype:
                            dtype = 'datetime'
                        elif 'date' in dtype:
                            dtype = 'date'
                        elif 'bool' in dtype:
                            dtype = 'boolean'
                        else:
                            dtype = 'str'

                        # Extract conditional_rules if present
                        conditional_rules = item.get('conditional_rules', [])
                        if not isinstance(conditional_rules, list):
                            conditional_rules = []

                        field = FieldDefinition(
                            field_name=item['field_name'],
                            data_type=dtype,
                            required=item.get('required', False),
                            description=item.get('description', ''),
                            min_value=item.get('min_value'),
                            max_value=item.get('max_value'),
                            allowed_values=item.get('allowed_values', []),
                            format_pattern=item.get('format_pattern'),
                            business_rules=item.get('business_rules', []),
                            conditional_rules=conditional_rules
                        )
                        fields.append(field)

        except LLMCallError:
            raise
        except Exception as e:
            logger.error(f"Error parsing LLM response ({len(response_text)} chars): {type(e).__name__}")
            raise LLMCallError(f"Could not parse LLM response: {type(e).__name__}") from e

        return fields

    def parse_responses_api_output(self, data: dict) -> str:
        """
        Parse Responses API output format (used by GPT-5 models).

        Responses API returns: {"output": [...]}
        where output is an array containing message objects
        """
        if not isinstance(data.get('output'), list):
            # Fallback for simple string output
            return str(data.get('output', ''))

        # Find message object in output array
        for item in data['output']:
            if item.get('type') == 'message' and item.get('content'):
                # Find text content in content array
                for content in item['content']:
                    if content.get('type') == 'output_text':
                        return content.get('text', '')

        # No text found, return empty
        logger.warning("No output_text found in Responses API output")
        return ''

    def extract_fields_using_responses_api(
        self,
        text_chunk: str,
        is_continuation: bool = False,
        deployment: Optional[str] = None,
    ) -> List[FieldDefinition]:
        """
        Extract field definitions using Responses API (GPT-5 models)

        Args:
            text_chunk: Dictionary text to extract fields from
            is_continuation: Whether this is a continuation chunk
            deployment: Optional per-call deployment override. Falls back to
                self.deployment (the configured default) when not provided,
                so the client can be constructed once at startup while still
                allowing callers (e.g. a UI model-selection dropdown) to pick
                a different deployment per request.
        """
        prompt = self.create_extraction_prompt(text_chunk, is_continuation)
        active_deployment = deployment or self.deployment
        max_output = max_output_tokens_for(active_deployment)

        print(f"[LLM] Sending to Responses API ({len(text_chunk)} chars, model={active_deployment}, max_output={max_output})...")
        response_text = self._call_responses_api(prompt, active_deployment, max_output)
        print(f"[LLM] Received response from Responses API")

        fields = self.parse_llm_response(response_text)

        print(f"[LLM] Extracted {len(fields)} fields from this chunk")
        logger.info(f"Extracted {len(fields)} fields from chunk")
        return fields

    def extract_fields_from_chunk(
        self,
        text_chunk: str,
        is_continuation: bool = False,
        deployment: Optional[str] = None,
    ) -> List[FieldDefinition]:
        """
        Extract field definitions using Responses API.

        All Azure OpenAI models now support Responses API.

        Args:
            text_chunk: Dictionary text to extract fields from
            is_continuation: Whether this is a continuation chunk
            deployment: Optional per-call deployment override (see
                extract_fields_using_responses_api for details).
        """
        return self.extract_fields_using_responses_api(text_chunk, is_continuation, deployment=deployment)

    def parse_dictionary(
        self,
        dictionary_text: str,
        max_fields: int = 1000,
        deployment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Parse entire data dictionary text into structured format

        Args:
            dictionary_text: Raw text from data dictionary
            max_fields: Maximum number of fields to extract (for safety)
            deployment: Optional Azure OpenAI deployment name to use for this
                call only (e.g. selected from a UI dropdown). When omitted,
                falls back to the client's configured default deployment
                (self.deployment / AZURE_OPENAI_DEPLOYMENT). Passing this
                does not mutate the client instance, so it's safe to reuse
                one LLMDictionaryParser across requests with different
                deployment choices.

        Returns:
            Dictionary with extracted schema and metadata
        """
        import time
        start_time = time.time()

        # Log to console for browser debugging
        print(f"[LLM] Starting dictionary parsing - {len(dictionary_text)} characters at {time.strftime('%H:%M:%S')}")
        logger.info(f"Parsing dictionary with {len(dictionary_text)} characters")

        # Check if we can process in a single call (more reliable than chunking)
        token_count = self.count_tokens(dictionary_text)
        print(f"[LLM] Estimated tokens: {token_count}")

        # If small enough, send entire dictionary in ONE call (avoids chunking errors)
        # Context window is 128k, leave margin for response
        if token_count < 80000:
            print(f"[LLM] ⚡ Using SINGLE-CALL mode (no chunking) - more reliable!")
            logger.info(f"Using single-call mode for {token_count} tokens")

            fields = self.extract_fields_from_chunk(dictionary_text, is_continuation=False, deployment=deployment)
            all_fields = fields

            elapsed_time = time.time() - start_time
            print(f"[LLM] Single-call parsing complete - extracted {len(all_fields)} fields in {elapsed_time:.2f} seconds")
            logger.info(f"Single-call parsing complete - {len(all_fields)} fields in {elapsed_time:.2f}s")

            # Convert to schema format
            schema = self.fields_to_schema(all_fields)

            return {
                "fields": [f.to_dict() for f in all_fields],
                "schema": schema,
                "metadata": {
                    "total_fields": len(all_fields),
                    "chunks_processed": 1,
                    "mode": "single-call",
                    "source": "LLM Parser",
                    "processing_time_seconds": elapsed_time
                }
            }

        # Otherwise, use chunking for very large dictionaries
        chunks = self.chunk_text(dictionary_text)
        print(f"[LLM] ⚠️ Using CHUNKED mode ({len(chunks)} chunks) - dictionary too large for single call")
        logger.info(f"Split into {len(chunks)} chunks")

        all_fields = []
        field_names_seen = set()

        for i, chunk in enumerate(chunks):
            if len(all_fields) >= max_fields:
                logger.warning(f"Reached maximum field limit of {max_fields}")
                break

            # Extract fields from chunk
            fields = self.extract_fields_from_chunk(chunk, is_continuation=(i > 0), deployment=deployment)

            # Deduplicate by field name
            for field in fields:
                if field.field_name not in field_names_seen:
                    all_fields.append(field)
                    field_names_seen.add(field.field_name)

            print(f"[LLM] Processed chunk {i+1}/{len(chunks)}, extracted {len(fields)} fields, total: {len(all_fields)}")
            logger.info(f"Processed chunk {i+1}/{len(chunks)}, total fields: {len(all_fields)}")

        # Convert to schema format
        schema = self.fields_to_schema(all_fields)

        elapsed_time = time.time() - start_time
        print(f"[LLM] Parsing complete - extracted {len(all_fields)} fields in {elapsed_time:.2f} seconds")
        logger.info(f"Parsing complete - extracted {len(all_fields)} fields in {elapsed_time:.2f} seconds")

        return {
            "fields": [f.to_dict() for f in all_fields],
            "schema": schema,
            "metadata": {
                "total_fields": len(all_fields),
                "chunks_processed": len(chunks),
                "mode": "chunked",
                "source": "LLM Parser",
                "processing_time_seconds": elapsed_time
            }
        }

    def fields_to_schema(self, fields: List[FieldDefinition]) -> Dict[str, Any]:
        """Convert field definitions to validation schema"""
        schema = {}

        for field in fields:
            field_schema = {
                "type": field.data_type,
                "required": field.required,
                "description": field.description
            }

            if field.min_value is not None:
                field_schema["min"] = field.min_value
            if field.max_value is not None:
                field_schema["max"] = field.max_value
            if field.allowed_values:
                field_schema["allowed_values"] = field.allowed_values
            if field.format_pattern:
                field_schema["format"] = field.format_pattern
            if field.business_rules:
                field_schema["rules"] = field.business_rules

            schema[field.field_name] = field_schema

        return schema

    def validate_data_with_llm(self, data_sample: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to validate data against schema and find complex issues

        Args:
            data_sample: Sample of data to validate (CSV or JSON string)
            schema: Validation schema

        Returns:
            Dictionary with validation results and issues found
        """
        prompt = f"""You are a data quality analyst. Analyze this data sample for quality issues based on the provided schema.
Look for:
1. Values that violate the schema rules
2. Semantic inconsistencies (e.g., end date before start date)
3. Likely data entry errors
4. Outliers or suspicious values
5. Missing required fields
6. Format violations

Schema:
```json
{json.dumps(schema, indent=2)[:2000]}
```

Data Sample:
```
{data_sample[:3000]}
```

Return a JSON object with:
- issues: Array of objects with keys: field, row, issue_type, description, severity
- summary: Overall data quality assessment
- recommendations: Array of suggested fixes or investigations
"""

        try:
            response_text = self._call_responses_api(
                prompt, self.deployment, max_output_tokens_for(self.deployment)
            )

            # Parse response
            try:
                json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                else:
                    return json.loads(response_text)
            except:
                return {
                    "issues": [],
                    "summary": "Could not parse validation results",
                    "recommendations": []
                }

        except LLMCallError:
            raise
        except Exception as e:
            logger.error(f"Error in LLM validation: {type(e).__name__}")
            raise LLMCallError(f"LLM validation failed: {type(e).__name__}") from e


# Utility function for testing
def test_parser():
    """Test the LLM parser with a sample dictionary"""
    parser = LLMDictionaryParser()

    # Test with simple CSV dictionary
    test_text = """
    Data Dictionary for Employee Records

    employee_id: Integer, Required, Range 1-999999
    Description: Unique identifier for each employee

    first_name: String, Required
    Description: Employee's first name

    salary: Decimal, Required, Range 30000-500000
    Description: Annual salary in USD

    department: String, Required
    Allowed Values: HR, Engineering, Sales, Marketing
    """

    result = parser.parse_dictionary(test_text)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    test_parser()