"""
Azure OpenAI LLM Client for Data Dictionary Parsing
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import re
from dotenv import load_dotenv
from openai import AzureOpenAI
import tiktoken

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        """Initialize Azure OpenAI client"""
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")

        if not all([self.endpoint, self.api_key]):
            raise ValueError("Azure OpenAI credentials not found in environment variables")

        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )

        # Token counting for gpt-4o-mini
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")

        logger.info(f"LLM client initialized with endpoint: {self.endpoint}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

    def chunk_text(self, text: str, max_tokens: int = 3000) -> List[str]:
        """Split text into chunks that fit within token limits"""
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
Extract field/column definitions from the following text and return them as JSON.

For each field, extract:
- field_name: The exact column/field name
- data_type: One of: int, float, str, date, datetime, boolean
- required: true if field is required/mandatory
- description: Field description
- min_value: Minimum value (if specified)
- max_value: Maximum value (if specified)
- allowed_values: List of allowed/valid values (if enumerated)
- format_pattern: Date/time format or regex pattern (if specified)
- business_rules: Any validation rules or business logic

Return a JSON array of field definitions. Be precise with field names.
Focus on extracting actual data fields, not metadata or headers.
"""

        continuation_prompt = """Continue extracting field definitions from this dictionary section.
Return only new fields not already processed."""

        prompt = continuation_prompt if is_continuation else base_prompt

        return f"""{prompt}

Dictionary text:
```
{text_chunk}
```

Return JSON array of field definitions:"""

    def parse_llm_response(self, response_text: str) -> List[FieldDefinition]:
        """Parse LLM response into FieldDefinition objects"""
        fields = []

        try:
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # Try to find JSON array directly
                json_text = response_text.strip()
                if not json_text.startswith('['):
                    # Look for array in the text
                    array_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                    if array_match:
                        json_text = array_match.group(0)

            # Parse JSON
            field_data = json.loads(json_text)

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

    def extract_fields_from_chunk(self, text_chunk: str, is_continuation: bool = False) -> List[FieldDefinition]:
        """Extract field definitions from a single text chunk using LLM"""
        try:
            prompt = self.create_extraction_prompt(text_chunk, is_continuation)

            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "You are a data dictionary parser. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=2000
            )

            response_text = response.choices[0].message.content
            fields = self.parse_llm_response(response_text)

            logger.info(f"Extracted {len(fields)} fields from chunk")
            return fields

        except Exception as e:
            logger.error(f"Error calling Azure OpenAI: {e}")
            return []

    def parse_dictionary(self, dictionary_text: str, max_fields: int = 1000) -> Dict[str, Any]:
        """
        Parse entire data dictionary text into structured format

        Args:
            dictionary_text: Raw text from data dictionary
            max_fields: Maximum number of fields to extract (for safety)

        Returns:
            Dictionary with extracted schema and metadata
        """
        logger.info(f"Parsing dictionary with {len(dictionary_text)} characters")

        # Chunk the text
        chunks = self.chunk_text(dictionary_text)
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

            logger.info(f"Processed chunk {i+1}/{len(chunks)}, total fields: {len(all_fields)}")

        # Convert to schema format
        schema = self.fields_to_schema(all_fields)

        return {
            "fields": [f.to_dict() for f in all_fields],
            "schema": schema,
            "metadata": {
                "total_fields": len(all_fields),
                "chunks_processed": len(chunks),
                "source": "LLM Parser"
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