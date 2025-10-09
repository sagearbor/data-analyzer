"""
Azure OpenAI LLM Client for Data Dictionary Parsing
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import re
import requests  # For Responses API calls
from dotenv import load_dotenv
import tiktoken

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

    def __post_init__(self):
        if self.allowed_values is None:
            self.allowed_values = []
        if self.business_rules is None:
            self.business_rules = []

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
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano")

        if not all([self.endpoint, self.api_key]):
            raise ValueError("Azure OpenAI credentials not found in environment variables")

        # Note: Using Responses API via requests library (not OpenAI SDK)
        # All Azure models now support /openai/v1/responses endpoint

        # Token counting for gpt-4o-mini
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")

        logger.info(f"LLM client initialized with endpoint: {self.endpoint}")

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

Return a JSON object with a "fields" array containing field definitions. Be precise with field names.
Extract ALL data fields you can identify, even if incomplete.
IMPORTANT for clinical/research dictionaries: Look for fields with patterns like _stop, _date, _dc, _id, etc.
IMPORTANT: For REDCap dictionaries, parse the "Choices, Calculations, OR Slider Labels" column to extract allowed_values.
"""

        continuation_prompt = """Continue extracting field definitions from this dictionary section.
Return only new fields not already processed as a JSON object with "fields" array."""

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
                    logger.error(f"Failed to repair JSON: {repair_error}")
                    logger.debug(f"Original text: {json_text[:500]}")
                    return fields

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

                        field = FieldDefinition(
                            field_name=item['field_name'],
                            data_type=dtype,
                            required=item.get('required', False),
                            description=item.get('description', ''),
                            min_value=item.get('min_value'),
                            max_value=item.get('max_value'),
                            allowed_values=item.get('allowed_values', []),
                            format_pattern=item.get('format_pattern'),
                            business_rules=item.get('business_rules', [])
                        )
                        fields.append(field)

        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.debug(f"Response text: {response_text[:500]}")

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

    def extract_fields_using_responses_api(self, text_chunk: str, is_continuation: bool = False) -> List[FieldDefinition]:
        """Extract field definitions using Responses API (GPT-5 models)"""
        try:
            prompt = self.create_extraction_prompt(text_chunk, is_continuation)

            # Responses API endpoint (no API version in path, no deployment in path)
            url = f"{self.endpoint.rstrip('/')}/openai/v1/responses"

            # Get max output tokens for this model
            deployment_lower = self.deployment.lower()
            max_output = MODEL_OUTPUT_LIMITS.get(deployment_lower, 128000)

            print(f"[LLM] Sending to Responses API ({len(text_chunk)} chars, max_output={max_output})...")

            # Build request for Responses API
            request_body = {
                "model": self.deployment,
                "input": prompt,  # String, not messages array!
                "max_output_tokens": max_output  # Not max_tokens!
            }

            # Add GPT-5 specific parameters
            request_body["reasoning"] = {"effort": "low"}  # low/medium/high
            request_body["text"] = {"verbosity": "medium"}  # minimal/medium/verbose

            response = requests.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "api-key": self.api_key
                },
                json=request_body,
                timeout=120
            )

            response.raise_for_status()
            data = response.json()

            print(f"[LLM] Received response from Responses API")

            # Parse output using Responses API format
            response_text = self.parse_responses_api_output(data)
            fields = self.parse_llm_response(response_text)

            print(f"[LLM] Extracted {len(fields)} fields from this chunk")
            logger.info(f"Extracted {len(fields)} fields from chunk")
            return fields

        except Exception as e:
            logger.error(f"Error calling Responses API: {e}")
            return []

    def extract_fields_from_chunk(self, text_chunk: str, is_continuation: bool = False) -> List[FieldDefinition]:
        """
        Extract field definitions using Responses API.

        All Azure OpenAI models now support Responses API.
        """
        return self.extract_fields_using_responses_api(text_chunk, is_continuation)

    def parse_dictionary(self, dictionary_text: str, max_fields: int = 1000) -> Dict[str, Any]:
        """
        Parse entire data dictionary text into structured format

        Args:
            dictionary_text: Raw text from data dictionary
            max_fields: Maximum number of fields to extract (for safety)

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

            fields = self.extract_fields_from_chunk(dictionary_text, is_continuation=False)
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
            fields = self.extract_fields_from_chunk(chunk, is_continuation=(i > 0))

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
        prompt = f"""Analyze this data sample for quality issues based on the provided schema.
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
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "You are a data quality analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )

            response_text = response.choices[0].message.content

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

        except Exception as e:
            logger.error(f"Error in LLM validation: {e}")
            return {
                "issues": [],
                "summary": f"Validation error: {str(e)}",
                "recommendations": []
            }


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