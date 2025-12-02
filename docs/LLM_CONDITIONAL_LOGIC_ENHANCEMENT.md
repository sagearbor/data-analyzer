# LLM-Based Conditional Logic Extraction Enhancement

## Overview

The Logic Validation Engine has been enhanced to extract conditional rules from **ANY dictionary format** using the LLM during initial parsing, making it versatile beyond the hardcoded REDCap/FHIR patterns.

## Key Design Principles

1. **LLM extracts logic from ANY format** - PDF, CSV, FHIR, REDCap, custom dictionaries
2. **LLM never sees actual data** - Only field definitions from the dictionary
3. **Structured format** - Returns ConditionalRule-compatible structure
4. **Defensive coding** - Graceful fallback for unparseable conditions

## Architecture

### Flow

```
Dictionary Text → LLM Parser → FieldDefinition (with conditional_rules)
                                        ↓
                               RuleExtractor._extract_llm_rules()
                                        ↓
                          _convert_natural_language_condition()
                                        ↓
                               ConditionalRule objects
                                        ↓
                                LogicValidator
```

### Priority Order

The `RuleExtractor.extract_rules_from_fields()` method now uses this priority:

1. **LLM-extracted conditional_rules** (FIRST - works with ANY format)
2. **Format-specific parsers** (REDCap/FHIR - as fallback/complement)
3. **Business rules text parsing** (legacy pattern matching)

## Implementation Details

### 1. Enhanced LLM Prompt

**File:** `/home/scb2/PROJECTS/gitRepos-wsl/data-analyzer/src/llm_client.py:128-187`

The prompt now instructs the LLM to extract conditional logic patterns:

- "skip if...", "required if...", "only when..."
- Cross-field dependencies
- Age/date constraints
- Value-based conditions

**Example Output:**
```json
{
  "fields": [
    {
      "field_name": "pregnancy_status",
      "data_type": "str",
      "conditional_rules": [
        {
          "rule_type": "skip_if",
          "condition_text": "gender is male",
          "action": "must_be_blank",
          "affected_fields": ["pregnancy_status"]
        }
      ]
    }
  ]
}
```

### 2. Updated Data Structure

**File:** `/home/scb2/PROJECTS/gitRepos-wsl/data-analyzer/src/llm_client.py:34-60`

Added `conditional_rules` field to `FieldDefinition`:

```python
@dataclass
class FieldDefinition:
    field_name: str
    data_type: str
    # ... other fields ...
    conditional_rules: List[Dict[str, Any]] = None  # NEW
```

### 3. Natural Language to Python Conversion

**File:** `/home/scb2/PROJECTS/gitRepos-wsl/data-analyzer/src/logic_engine.py:941-1019`

The `_convert_natural_language_condition()` method handles:

| Natural Language | Python Expression |
|-----------------|-------------------|
| "gender is male" | `str(row.get('gender', '')).lower() in ['male', 'm', '1']` |
| "age >= 18" | `int(row.get('age', 0)) >= 18` |
| "pregnant is yes" | `str(row.get('pregnant', '')).lower() in ['yes', 'y', '1', 'true']` |
| "treatment_arm is control" | `str(row.get('treatment_arm', '')).lower() == 'control'` |
| "diagnosis contains cancer" | `'cancer' in str(row.get('diagnosis', '')).lower()` |

**Defensive Coding:** Returns `"False"` for unparseable patterns (skip validation safely).

### 4. LLM Rule Extraction

**File:** `/home/scb2/PROJECTS/gitRepos-wsl/data-analyzer/src/logic_engine.py:1021-1089`

The `_extract_llm_rules()` method:

1. Extracts `conditional_rules` array from field definition
2. Converts natural language conditions to Python
3. Creates `ConditionalRule` objects with:
   - `confidence`: 0.85 (slightly lower than explicit rules)
   - `severity`: "error" for skip_if/required_if, "warning" for show_if/allowed_if
   - `source`: "LLM extraction: {condition_text}"

### 5. Prioritization in RuleExtractor

**File:** `/home/scb2/PROJECTS/gitRepos-wsl/data-analyzer/src/logic_engine.py:639-685`

Modified `extract_rules_from_fields()`:

```python
# PRIORITY 1: Try LLM-extracted conditional_rules FIRST
llm_rules = self._extract_llm_rules(field)
if llm_rules:
    rules.extend(llm_rules)
    logger.debug(f"Field '{field_name}': Using {len(llm_rules)} LLM-extracted rule(s)")

# PRIORITY 2: Format-specific parsers as fallback/complement
if format_type.lower() == "redcap":
    format_rules = self._extract_redcap_rules(field)
    rules.extend(format_rules)
```

## Supported Patterns

### Equality Checks
- "field is value" → `str(row.get('field', '')).lower() == 'value'`
- "field = value" → Same as above

### Special Values
- Male: `['male', 'm', '1']`
- Female: `['female', 'f', '2']`
- Yes: `['yes', 'y', '1', 'true']`
- No: `['no', 'n', '0', 'false']`

### Numeric Comparisons
- "age >= 18" → `int(row.get('age', 0)) >= 18`
- "score > 50" → `int(row.get('score', 0)) > 50`
- "count <= 10" → `int(row.get('count', 0)) <= 10`

### String Operations
- "field contains value" → `'value' in str(row.get('field', '')).lower()`
- "field != value" → `str(row.get('field', '')).lower() != 'value'`

## Example Usage

### Dictionary Text (ANY Format)

```
Field: pregnancy_status
Type: String
Description: Pregnancy status of participant
Rules: If male, skip this field

Field: alcohol_consumption
Type: Integer
Description: Number of alcoholic drinks per week
Rules: Only if age >= 18
```

### LLM Extraction

The LLM extracts:

```json
{
  "fields": [
    {
      "field_name": "pregnancy_status",
      "conditional_rules": [
        {
          "rule_type": "skip_if",
          "condition_text": "gender is male",
          "action": "must_be_blank",
          "affected_fields": ["pregnancy_status"]
        }
      ]
    },
    {
      "field_name": "alcohol_consumption",
      "conditional_rules": [
        {
          "rule_type": "show_if",
          "condition_text": "age >= 18",
          "action": "must_be_filled",
          "affected_fields": ["alcohol_consumption"]
        }
      ]
    }
  ]
}
```

### Conversion to ConditionalRule

```python
ConditionalRule(
    rule_id="pregnancy_status_llm_0",
    rule_type="skip_if",
    condition="str(row.get('gender', '')).lower() in ['male', 'm', '1']",
    action="must_be_blank",
    affected_fields=["pregnancy_status"],
    description="LLM-extracted: gender is male",
    source="LLM extraction: gender is male",
    severity="error",
    confidence=0.85
)
```

## Testing

### Test Coverage

**File:** `/home/scb2/PROJECTS/gitRepos-wsl/data-analyzer/tests/test_llm_conditional_extraction.py`

- 16 new tests for LLM conditional extraction
- All tests pass
- 47 existing logic_engine tests still pass
- **Total: 63 tests passing**

### Key Test Cases

1. Natural language conversion for common patterns
2. Multiple rules per field
3. Malformed data handling (graceful degradation)
4. Priority over format-specific parsers
5. Confidence and severity scoring

### Run Tests

```bash
# Test new LLM extraction functionality
pytest tests/test_llm_conditional_extraction.py -v

# Test full logic engine (including integration)
pytest tests/test_logic_engine.py -v

# Test everything
pytest tests/test_logic_engine.py tests/test_llm_conditional_extraction.py -v
```

## Advantages

1. **Works with ANY dictionary format** - Not limited to REDCap/FHIR
2. **No manual pattern coding** - LLM learns new patterns automatically
3. **Backwards compatible** - REDCap/FHIR parsers still work as fallback
4. **Defensive** - Gracefully handles unparseable conditions
5. **Extensible** - Easy to add new natural language patterns

## Limitations

1. **LLM accuracy** - Confidence set to 0.85 (slightly lower than explicit rules)
2. **Pattern coverage** - May not recognize all obscure conditional patterns
3. **Fallback required** - Still returns "False" for unparseable conditions

## Future Enhancements

1. Add more natural language patterns to `_convert_natural_language_condition()`
2. Use LLM to convert complex conditions (if regex patterns fail)
3. Track LLM extraction accuracy and adjust confidence scores
4. Add support for compound conditions (AND/OR)
5. Generate test data based on extracted rules

## Files Modified

1. `/home/scb2/PROJECTS/gitRepos-wsl/data-analyzer/src/llm_client.py`
   - Lines 34-60: Added `conditional_rules` to `FieldDefinition`
   - Lines 128-187: Enhanced prompt for conditional logic extraction
   - Lines 280-295: Updated parser to extract conditional_rules

2. `/home/scb2/PROJECTS/gitRepos-wsl/data-analyzer/src/logic_engine.py`
   - Lines 639-685: Modified `extract_rules_from_fields()` to prioritize LLM rules
   - Lines 941-1019: Added `_convert_natural_language_condition()` method
   - Lines 1021-1089: Added `_extract_llm_rules()` method

3. `/home/scb2/PROJECTS/gitRepos-wsl/data-analyzer/tests/test_llm_conditional_extraction.py`
   - New file: 16 comprehensive tests for LLM extraction

## Conclusion

The Logic Validation Engine now uses LLM-based extraction as the PRIMARY method for identifying conditional rules, making it versatile enough to work with ANY dictionary format while maintaining backward compatibility with REDCap/FHIR-specific parsers.
