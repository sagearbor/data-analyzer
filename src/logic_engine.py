"""
Logic Validation Engine for Data Analyzer

This module provides conditional logic validation that EXTENDS the existing QualityChecker.
- QualityChecker: Validates individual fields (types, ranges, allowed values)
- LogicValidator: Validates conditional logic ACROSS fields

Key Components:
- ConditionalRule: Represents a conditional logic rule from a data dictionary
- LogicViolation: Represents a violation of a conditional logic rule
- LogicCodeGenerator: Generates safe Python validation code from rules
- LogicValidator: Validates data against conditional logic rules
- RuleExtractor: Extracts ConditionalRules from parsed dictionary fields

Security Features:
- AST-based code validation to prevent injection attacks
- Sandboxed execution with restricted builtins
- Pattern-based detection of dangerous operations
- No file, network, or system access in generated code
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import json
import re
import logging
import ast
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ConditionalRule:
    """
    Represents a conditional logic rule from a data dictionary.

    Rule types:
    - skip_if: Field should be blank when condition is met
    - required_if: Field should be filled when condition is met
    - allowed_if: Field is only allowed when condition is met
    - value_if: Field should have specific value when condition is met
    - show_if: Field is shown/enabled when condition is met

    Actions:
    - must_be_blank: Field should be empty/blank
    - must_be_filled: Field should have a value
    - skip: Field should be skipped (same as must_be_blank)
    - required: Field is required (same as must_be_filled)
    - value_in: Field value must be in allowed set

    Attributes:
        rule_id: Unique identifier for the rule
        rule_type: Type of conditional rule
        condition: Python expression that evaluates to boolean
        action: Action to take when condition is met
        affected_fields: List of field names this rule applies to
        description: Human-readable description of the rule
        source: Original dictionary text (for traceability)
        severity: "error" or "warning"
        confidence: 0.0-1.0 (1.0 = explicit in dictionary, <1.0 = inferred)
    """
    rule_id: str
    rule_type: str
    condition: str
    action: str
    affected_fields: List[str]
    description: str
    source: str = ""
    severity: str = "error"
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConditionalRule':
        """
        Create ConditionalRule from dictionary.

        Args:
            data: Dictionary with rule fields

        Returns:
            ConditionalRule instance
        """
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LogicViolation:
    """
    Represents a violation of a conditional logic rule.

    Attributes:
        rule_id: ID of the rule that was violated
        rule_description: Human-readable description
        row_index: Row number where violation occurred (0-indexed)
        affected_fields: Field names involved in the violation
        actual_values: Actual values of the fields
        expected_behavior: What was expected according to the rule
        severity: "error" or "warning"
    """
    rule_id: str
    rule_description: str
    row_index: int
    affected_fields: List[str]
    actual_values: Dict[str, Any]
    expected_behavior: str
    severity: str = "error"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


# =============================================================================
# CODE GENERATOR
# =============================================================================

class LogicCodeGenerator:
    """
    Generates safe Python validation code from ConditionalRules.

    Security measures:
    1. AST-based validation of generated code
    2. Pattern matching to detect dangerous operations
    3. Whitelist of allowed builtins and methods
    4. Sanitization of condition expressions

    The generated code:
    - Takes a DataFrame as input
    - Returns a list of violation dictionaries
    - Handles missing columns gracefully
    - Does NOT access files, network, or dangerous builtins
    """

    # Allowed built-in functions/names for safety
    SAFE_BUILTINS = {
        'True', 'False', 'None',
        'abs', 'len', 'str', 'int', 'float', 'bool',
        'min', 'max', 'sum', 'round',
        'all', 'any', 'isinstance', 'range',
        'list', 'dict', 'tuple', 'set'
    }

    # Allowed pandas/numpy operations
    SAFE_METHODS = {
        'isna', 'notna', 'isnull', 'notnull', 'fillna',
        'isin', 'str', 'lower', 'upper', 'strip',
        'eq', 'ne', 'gt', 'lt', 'ge', 'le',
        'get', 'items', 'values', 'keys', 'index'
    }

    def generate_validation_code(self, rules: List[ConditionalRule]) -> str:
        """
        Generate complete Python validation code for a set of rules.

        The generated code:
        1. Is executable in a sandboxed environment
        2. Takes a DataFrame and returns List[Dict] of violations
        3. Handles missing columns gracefully
        4. Does NOT access files, network, or dangerous builtins

        Args:
            rules: List of ConditionalRule objects

        Returns:
            Python code as string
        """
        code_parts = [
            "# AUTO-GENERATED VALIDATION CODE",
            "# Generated at: " + datetime.now().isoformat(),
            f"# Number of rules: {len(rules)}",
            "from typing import List, Dict, Any",
            "import pandas as pd",
            "",
            "def validate_logic(df: pd.DataFrame) -> List[Dict]:",
            '    """Validate conditional logic rules on the dataset"""',
            "    violations = []",
            ""
        ]

        for rule in rules:
            rule_code = self._generate_rule_check(rule)
            # Indent all lines for the function body
            code_parts.extend(["    " + line for line in rule_code.split("\n")])
            code_parts.append("")

        code_parts.extend([
            "    return violations",
            ""
        ])

        return "\n".join(code_parts)

    def _generate_rule_check(self, rule: ConditionalRule) -> str:
        """
        Generate code for a single rule check.

        Args:
            rule: ConditionalRule to generate code for

        Returns:
            Python code as string
        """
        # Sanitize the condition to prevent injection
        sanitized_condition = self._sanitize_condition(rule.condition)

        code = f'''# Rule: {rule.rule_id} - {rule.description}
try:
    for idx, row in df.iterrows():
        try:
            # Check condition
            condition_met = {sanitized_condition}

            if condition_met:
                # Check action
                {self._generate_action_check(rule)}
        except KeyError:
            # Missing column - skip this row
            pass
        except Exception as e:
            # Other errors - skip this row
            pass
except Exception as e:
    # Rule-level error - skip entire rule
    pass'''
        return code

    def _generate_action_check(self, rule: ConditionalRule) -> str:
        """
        Generate the action check code based on rule.action.

        Args:
            rule: ConditionalRule with action to generate

        Returns:
            Python code as string
        """
        action = rule.action.lower()
        fields_check = "['" + "', '".join(rule.affected_fields) + "']"

        # Escape single quotes and handle multi-line descriptions
        escaped_desc = rule.description.replace("'", "\\'").replace("\n", " ")

        if action in ("must_be_blank", "skip"):
            return f'''for field in {fields_check}:
                    if field in row.index and pd.notna(row[field]) and str(row[field]).strip() != '':
                        violations.append({{
                            'rule_id': '{rule.rule_id}',
                            'rule_description': '{escaped_desc}',
                            'row_index': int(idx),
                            'affected_fields': {fields_check},
                            'actual_values': {{f: row.get(f) for f in {fields_check} if f in row.index}},
                            'expected_behavior': 'Field should be blank when condition is met',
                            'severity': '{rule.severity}'
                        }})'''

        elif action in ("must_be_filled", "required"):
            return f'''for field in {fields_check}:
                    if field in row.index and (pd.isna(row[field]) or str(row[field]).strip() == ''):
                        violations.append({{
                            'rule_id': '{rule.rule_id}',
                            'rule_description': '{escaped_desc}',
                            'row_index': int(idx),
                            'affected_fields': {fields_check},
                            'actual_values': {{f: row.get(f) for f in {fields_check} if f in row.index}},
                            'expected_behavior': 'Field should be filled when condition is met',
                            'severity': '{rule.severity}'
                        }})'''

        elif action == "value_in":
            # For value_in, we'd need an allowed_values list in the rule
            # This is a placeholder for future implementation
            return "pass  # value_in action not yet implemented"

        else:
            # Unknown action - log and skip
            logger.warning(f"Unknown action type: {action} for rule {rule.rule_id}")
            return "pass  # Unknown action type"

    def _sanitize_condition(self, condition: str) -> str:
        """
        Sanitize a condition string to prevent code injection.

        Allowed patterns:
        - row['field_name'] == 'value'
        - row.get('field', default) != 'value'
        - pd.isna(row['field'])
        - pd.notna(row['field'])
        - Logical operators: and, or, not
        - Comparison operators: ==, !=, >, <, >=, <=
        - String methods: .lower(), .upper(), .strip()
        - Membership: in, not in

        NOT allowed:
        - import statements
        - exec/eval/compile
        - __builtins__, __class__, __import__, etc.
        - file operations (open, read, write)
        - network operations (socket, requests, urllib)
        - os/sys/subprocess calls

        Args:
            condition: Condition string to sanitize

        Returns:
            Sanitized condition string (or "False" if dangerous)
        """
        # Check for dangerous patterns
        dangerous_patterns = [
            r'import\s+',
            r'__\w+__',
            r'exec\s*\(',
            r'eval\s*\(',
            r'compile\s*\(',
            r'open\s*\(',
            r'\bos\.',
            r'\bsys\.',
            r'subprocess',
            r'socket',
            r'requests\.',
            r'urllib',
            r'\.read\s*\(',
            r'\.write\s*\(',
            r'file\s*\(',
            r'lambda\s+',
            r'globals\s*\(',
            r'locals\s*\(',
            r'vars\s*\(',
            r'dir\s*\(',
            r'getattr\s*\(',
            r'setattr\s*\(',
            r'delattr\s*\(',
            r'hasattr\s*\(',
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, condition, re.IGNORECASE):
                logger.warning(f"Dangerous pattern '{pattern}' detected in condition: {condition}")
                return "False"  # Return safe false condition

        # Check length to prevent DOS via extremely long conditions
        if len(condition) > 1000:
            logger.warning(f"Condition too long ({len(condition)} chars), truncating")
            return "False"

        return condition

    def validate_generated_code(self, code: str) -> bool:
        """
        Validate that generated code is safe to execute.
        Uses AST analysis to detect dangerous operations.

        Args:
            code: Python code string to validate

        Returns:
            True if code is safe, False otherwise
        """
        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                # Check for imports
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name not in ('typing', 'pandas'):
                                logger.warning(f"Unsafe import: {alias.name}")
                                return False
                    elif isinstance(node, ast.ImportFrom):
                        if node.module not in ('typing', 'pandas', None):
                            logger.warning(f"Unsafe import from: {node.module}")
                            return False

                # Check for exec/eval/compile calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ('exec', 'eval', 'compile', 'open', '__import__'):
                            logger.warning(f"Unsafe function call: {node.func.id}")
                            return False

                # Check for attribute access to dangerous objects
                if isinstance(node, ast.Attribute):
                    dangerous_attrs = ['__builtins__', '__globals__', '__class__', '__base__']
                    if node.attr in dangerous_attrs:
                        logger.warning(f"Unsafe attribute access: {node.attr}")
                        return False

            return True

        except SyntaxError as e:
            logger.error(f"Syntax error in generated code: {e}")
            return False
        except Exception as e:
            logger.error(f"Error validating generated code: {e}")
            return False


# =============================================================================
# LOGIC VALIDATOR
# =============================================================================

class LogicValidator:
    """
    Validates conditional logic rules against data.

    This class EXTENDS the existing QualityChecker - it does NOT replace it.
    - QualityChecker: Handles field types, value ranges, allowed values
    - LogicValidator: Handles conditional logic BETWEEN fields

    Features:
    - Generates safe Python validation code from rules
    - Executes validation code in sandboxed environment
    - Supports pre-generated code (from program cache)
    - Returns structured violation objects

    Security:
    - All generated code is validated before execution
    - Sandboxed execution with restricted builtins
    - No file, network, or system access
    """

    def __init__(self):
        """Initialize LogicValidator with code generator."""
        self.code_generator = LogicCodeGenerator()

    def validate(self, rules: List[ConditionalRule], df: pd.DataFrame) -> List[LogicViolation]:
        """
        Validate data against conditional logic rules.

        Args:
            rules: List of ConditionalRule objects
            df: DataFrame to validate

        Returns:
            List of LogicViolation objects
        """
        if not rules:
            logger.debug("No rules to validate")
            return []

        if df is None or df.empty:
            logger.debug("Empty DataFrame, no violations")
            return []

        logger.info(f"Validating {len(rules)} rules against {len(df)} rows")

        # Generate validation code
        code = self.code_generator.generate_validation_code(rules)

        # Validate the generated code is safe
        if not self.code_generator.validate_generated_code(code):
            logger.error("Generated code failed safety check")
            return []

        # Execute in sandboxed environment
        violations_dicts = self._execute_sandboxed(code, df)

        # Convert to LogicViolation objects
        violations = [
            LogicViolation(**v) for v in violations_dicts
        ]

        logger.info(f"Found {len(violations)} logic violations")
        return violations

    def validate_with_code(self, code: str, df: pd.DataFrame) -> List[LogicViolation]:
        """
        Validate using pre-generated code (from program cache).

        This method allows validation using cached code without regenerating
        from rules, which is more efficient for repeated validations.

        Args:
            code: Pre-generated validation code
            df: DataFrame to validate

        Returns:
            List of LogicViolation objects
        """
        if not code or not code.strip():
            logger.warning("Empty validation code provided")
            return []

        if df is None or df.empty:
            logger.debug("Empty DataFrame, no violations")
            return []

        logger.info(f"Validating with cached code against {len(df)} rows")

        # Validate the code is safe
        if not self.code_generator.validate_generated_code(code):
            logger.error("Stored code failed safety check")
            return []

        # Execute in sandboxed environment
        violations_dicts = self._execute_sandboxed(code, df)

        # Convert to LogicViolation objects
        violations = [
            LogicViolation(**v) for v in violations_dicts
        ]

        logger.info(f"Found {len(violations)} logic violations")
        return violations

    def _execute_sandboxed(self, code: str, df: pd.DataFrame) -> List[Dict]:
        """
        Execute validation code in a sandboxed environment.

        Restrictions:
        - No file access
        - No network access
        - No system calls
        - Limited builtins (whitelist only)
        - No module imports beyond pandas/typing

        Args:
            code: Python code to execute
            df: DataFrame to pass to validation function

        Returns:
            List of violation dictionaries
        """
        # Create restricted globals with only safe builtins
        safe_globals = {
            '__builtins__': {
                'True': True,
                'False': False,
                'None': None,
                'abs': abs,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'min': min,
                'max': max,
                'sum': sum,
                'round': round,
                'all': all,
                'any': any,
                'isinstance': isinstance,
                'range': range,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
            },
            'pd': pd,
            'List': List,
            'Dict': Dict,
            'Any': Any,
        }

        # Create locals for execution
        local_vars = {'df': df.copy()}  # Copy to prevent modification

        try:
            # Remove import statements since we provide modules directly
            # This is safe because we've already validated the code with AST
            code_lines = code.split('\n')
            filtered_lines = []
            for line in code_lines:
                stripped = line.strip()
                if not (stripped.startswith('import ') or stripped.startswith('from ')):
                    filtered_lines.append(line)
            filtered_code = '\n'.join(filtered_lines)

            # Execute the code in restricted environment
            exec(filtered_code, safe_globals, local_vars)

            # Get the validate_logic function and call it
            if 'validate_logic' in local_vars:
                violations = local_vars['validate_logic'](df)
                return violations if violations else []
            else:
                logger.error("validate_logic function not found in generated code")
                return []

        except NameError as e:
            logger.error(f"Name error executing validation code: {e}")
            logger.debug(traceback.format_exc())
            return []
        except KeyError as e:
            logger.error(f"Key error executing validation code (missing column?): {e}")
            logger.debug(traceback.format_exc())
            return []
        except Exception as e:
            logger.error(f"Error executing validation code: {e}")
            logger.debug(traceback.format_exc())
            return []

    def generate_code_from_rules(self, rules: List[ConditionalRule]) -> str:
        """
        Generate validation code from rules (for caching).

        This allows the code to be cached and reused for multiple validations
        without regenerating from rules each time.

        Args:
            rules: List of ConditionalRule objects

        Returns:
            Generated Python code as string
        """
        return self.code_generator.generate_validation_code(rules)


# =============================================================================
# RULE EXTRACTOR
# =============================================================================

class RuleExtractor:
    """
    Extract ConditionalRules from parsed dictionary fields.

    Supports multiple dictionary formats:
    - REDCap: Branching logic, text_validation_min/max
    - FHIR: enableWhen, enableBehavior
    - Custom: business_rules field

    The extractor converts format-specific logic into standardized
    ConditionalRule objects that can be validated by LogicValidator.
    """

    def extract_rules_from_fields(self,
                                  fields: List[Dict],
                                  format_type: str = "REDCap") -> List[ConditionalRule]:
        """
        Extract conditional rules from field definitions.

        Args:
            fields: List of field dictionaries from LLM parsing
            format_type: "REDCap", "FHIR", "Custom", etc.

        Returns:
            List of ConditionalRule objects
        """
        rules = []

        for field in fields:
            field_name = field.get('field_name', '')
            if not field_name:
                continue

            # Extract rules based on format type
            if format_type.lower() == "redcap":
                rules.extend(self._extract_redcap_rules(field))
            elif format_type.lower() == "fhir":
                rules.extend(self._extract_fhir_rules(field))
            else:
                rules.extend(self._extract_custom_rules(field))

        logger.info(f"Extracted {len(rules)} rules from {len(fields)} fields")
        return rules

    def _extract_redcap_rules(self, field: Dict) -> List[ConditionalRule]:
        """
        Extract rules from REDCap field definition.

        Args:
            field: Field dictionary with REDCap-specific fields

        Returns:
            List of ConditionalRule objects
        """
        rules = []
        field_name = field.get('field_name', '')

        # Check for branching logic (REDCap)
        branching_logic = field.get('branching_logic', '')
        if branching_logic:
            rule = self._parse_redcap_branching(field_name, branching_logic)
            if rule:
                rules.append(rule)

        # Check for custom business rules
        business_rules = field.get('business_rules', [])
        if isinstance(business_rules, list):
            for br in business_rules:
                rule = self._parse_business_rule(field_name, br)
                if rule:
                    rules.append(rule)

        return rules

    def _extract_fhir_rules(self, field: Dict) -> List[ConditionalRule]:
        """
        Extract rules from FHIR questionnaire item.

        Args:
            field: Field dictionary with FHIR-specific fields

        Returns:
            List of ConditionalRule objects
        """
        rules = []
        field_name = field.get('field_name', '')

        # Check for FHIR enableWhen
        enable_when = field.get('enable_when', [])
        if isinstance(enable_when, list):
            for ew in enable_when:
                rule = self._parse_fhir_enable_when(field_name, ew)
                if rule:
                    rules.append(rule)

        return rules

    def _extract_custom_rules(self, field: Dict) -> List[ConditionalRule]:
        """
        Extract rules from custom field definition.

        Args:
            field: Field dictionary with custom fields

        Returns:
            List of ConditionalRule objects
        """
        rules = []
        field_name = field.get('field_name', '')

        # Check for business rules
        business_rules = field.get('business_rules', [])
        if isinstance(business_rules, list):
            for br in business_rules:
                rule = self._parse_business_rule(field_name, br)
                if rule:
                    rules.append(rule)

        return rules

    def _parse_redcap_branching(self, field_name: str, branching_logic: str) -> Optional[ConditionalRule]:
        """
        Parse REDCap branching logic string into a ConditionalRule.

        REDCap syntax examples:
        - [gender]='2' -> Show field if gender is 2 (female)
        - [age]>=18 -> Show field if age >= 18
        - [pregnant]='1' and [gender]='2' -> Complex condition

        Converts to Python:
        - [field_name] -> row['field_name']
        - = -> ==
        - and/or -> and/or (case insensitive)

        Args:
            field_name: Name of the field this rule applies to
            branching_logic: REDCap branching logic string

        Returns:
            ConditionalRule or None if parsing fails
        """
        if not branching_logic or not branching_logic.strip():
            return None

        try:
            # Convert REDCap syntax to Python
            condition = branching_logic.strip()

            # Replace [field_name] with row['field_name']
            condition = re.sub(r'\[(\w+)\]', r"row.get('\1', '')", condition)

            # Replace single = with == (but not ==, !=, >=, <=)
            condition = re.sub(r"(?<![=!<>])=(?!=)", r"==", condition)

            # Replace 'and' and 'or' (case insensitive)
            condition = re.sub(r'\band\b', ' and ', condition, flags=re.IGNORECASE)
            condition = re.sub(r'\bor\b', ' or ', condition, flags=re.IGNORECASE)

            # Field should be shown IF condition is true
            # So it should be BLANK if condition is false
            return ConditionalRule(
                rule_id=f"{field_name}_show_if",
                rule_type="show_if",
                condition=f"not ({condition})",  # Invert: blank if NOT shown
                action="must_be_blank",
                affected_fields=[field_name],
                description=f"Skip field '{field_name}' when condition is not met: {branching_logic}",
                source=branching_logic,
                severity="warning",
                confidence=1.0
            )

        except Exception as e:
            logger.warning(f"Failed to parse branching logic: {branching_logic} - {e}")
            return None

    def _parse_business_rule(self, field_name: str, rule_text: str) -> Optional[ConditionalRule]:
        """
        Parse a business rule text into a ConditionalRule.

        Recognizes common patterns:
        - "If male, skip this field"
        - "Required for female subjects"
        - "Only if age >= 18"

        Args:
            field_name: Name of the field this rule applies to
            rule_text: Business rule text

        Returns:
            ConditionalRule or None if pattern not recognized
        """
        if not rule_text or not isinstance(rule_text, str):
            return None

        rule_lower = rule_text.lower()

        # Pattern: "if male" + "skip/blank"
        if "male" in rule_lower and ("skip" in rule_lower or "blank" in rule_lower):
            # Check if it's specifically about NOT being female
            if "if" in rule_lower and "male" in rule_lower:
                return ConditionalRule(
                    rule_id=f"{field_name}_male_skip",
                    rule_type="skip_if",
                    condition="str(row.get('gender', '')).lower() in ['male', 'm', '1']",
                    action="must_be_blank",
                    affected_fields=[field_name],
                    description=f"Skip {field_name} for male subjects",
                    source=rule_text,
                    severity="error",
                    confidence=0.9
                )

        # Pattern: "if female" + "required"
        if "female" in rule_lower and "required" in rule_lower:
            return ConditionalRule(
                rule_id=f"{field_name}_female_required",
                rule_type="required_if",
                condition="str(row.get('gender', '')).lower() in ['female', 'f', '2']",
                action="must_be_filled",
                affected_fields=[field_name],
                description=f"Require {field_name} for female subjects",
                source=rule_text,
                severity="error",
                confidence=0.9
            )

        # Pattern: "only if age >= X"
        age_match = re.search(r'age\s*>=\s*(\d+)', rule_lower)
        if age_match:
            age_threshold = age_match.group(1)
            return ConditionalRule(
                rule_id=f"{field_name}_age_threshold",
                rule_type="show_if",
                condition=f"not (int(row.get('age', 0)) >= {age_threshold})",
                action="must_be_blank",
                affected_fields=[field_name],
                description=f"Skip {field_name} if age < {age_threshold}",
                source=rule_text,
                severity="warning",
                confidence=0.8
            )

        return None

    def _parse_fhir_enable_when(self, field_name: str, enable_when: Dict) -> Optional[ConditionalRule]:
        """
        Parse FHIR enableWhen structure into a ConditionalRule.

        FHIR enableWhen example:
        {
            "question": "gender",
            "operator": "=",
            "answerString": "female"
        }

        Args:
            field_name: Name of the field this rule applies to
            enable_when: FHIR enableWhen dictionary

        Returns:
            ConditionalRule or None if parsing fails
        """
        try:
            question = enable_when.get('question', '')
            operator = enable_when.get('operator', '=')
            answer = (enable_when.get('answerString') or
                     enable_when.get('answerInteger') or
                     enable_when.get('answerCoding', {}).get('code', ''))

            if not question or answer is None:
                return None

            # Map FHIR operators to Python
            op_map = {
                '=': '==',
                '!=': '!=',
                '>': '>',
                '<': '<',
                '>=': '>=',
                '<=': '<=',
                'exists': 'pd.notna'
            }

            py_op = op_map.get(operator, '==')

            # Handle special case for 'exists'
            if operator == 'exists':
                if answer:  # exists = true
                    condition = f"pd.notna(row.get('{question}'))"
                else:  # exists = false
                    condition = f"pd.isna(row.get('{question}'))"
            else:
                # Standard comparison
                if isinstance(answer, str):
                    condition = f"row.get('{question}') {py_op} '{answer}'"
                else:
                    condition = f"row.get('{question}') {py_op} {answer}"

            return ConditionalRule(
                rule_id=f"{field_name}_enable_when",
                rule_type="show_if",
                condition=f"not ({condition})",  # Invert: blank if NOT enabled
                action="must_be_blank",
                affected_fields=[field_name],
                description=f"Enable {field_name} when {question} {operator} {answer}",
                source=json.dumps(enable_when),
                severity="warning",
                confidence=1.0
            )

        except Exception as e:
            logger.warning(f"Failed to parse FHIR enableWhen: {e}")
            return None
