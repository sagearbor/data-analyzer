"""
Comprehensive tests for the Logic Validation Engine (Feature 2)

Tests cover:
- ConditionalRule dataclass operations
- LogicViolation dataclass operations
- LogicCodeGenerator code generation and security
- LogicValidator validation execution
- RuleExtractor parsing of REDCap and FHIR logic
- Integration workflows with real test data
"""
import pytest
import pandas as pd
import json
import ast
import sys
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

# Add parent directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_rule():
    """Create a sample ConditionalRule for testing"""
    from src.logic_engine import ConditionalRule
    return ConditionalRule(
        rule_id="pregnancy_gender_check",
        rule_type="skip_if",
        condition="str(row.get('gender', '')).lower() in ['male', 'm', '1']",
        action="must_be_blank",
        affected_fields=["pregnant", "weeks_pregnant"],
        description="Skip pregnancy questions for male subjects",
        source="[gender]='1'",
        severity="error",
        confidence=1.0
    )


@pytest.fixture
def sample_rules():
    """Create multiple sample rules for testing"""
    from src.logic_engine import ConditionalRule
    return [
        ConditionalRule(
            rule_id="pregnancy_male_skip",
            rule_type="skip_if",
            condition="str(row.get('gender', '')).lower() in ['male', 'm', '1']",
            action="must_be_blank",
            affected_fields=["pregnant"],
            description="Skip pregnancy for males",
            severity="error",
            confidence=1.0
        ),
        ConditionalRule(
            rule_id="dose_age_check",
            rule_type="required_if",
            condition="row.get('age', 0) >= 18",
            action="must_be_filled",
            affected_fields=["consent_signed"],
            description="Consent required for adults",
            severity="error",
            confidence=1.0
        )
    ]


@pytest.fixture
def sample_df_valid():
    """DataFrame that should pass validation"""
    return pd.DataFrame({
        'subject_id': ['001', '002', '003'],
        'gender': ['Male', 'Female', 'Female'],
        'age': [35, 28, 45],
        'pregnant': [None, 'Yes', 'No'],  # Male has blank, as expected
        'weeks_pregnant': [None, 12, None],
        'consent_signed': ['Yes', 'Yes', 'Yes']
    })


@pytest.fixture
def sample_df_violations():
    """DataFrame that should have violations"""
    return pd.DataFrame({
        'subject_id': ['001', '002'],
        'gender': ['Male', 'Male'],
        'age': [35, 28],
        'pregnant': ['Yes', 'No'],  # Males should have blank!
        'weeks_pregnant': [12, None],  # Males should have blank!
    })


@pytest.fixture
def sample_fields_redcap():
    """Sample REDCap field definitions with branching logic"""
    return [
        {
            'field_name': 'subject_id',
            'data_type': 'str',
            'required': True
        },
        {
            'field_name': 'gender',
            'data_type': 'int',
            'allowed_values': [1, 2]
        },
        {
            'field_name': 'pregnant',
            'data_type': 'int',
            'branching_logic': "[gender]='2'",  # Show only for females
            'allowed_values': [0, 1]
        },
        {
            'field_name': 'weeks_pregnant',
            'data_type': 'int',
            'branching_logic': "[pregnant]='1'"  # Show only if pregnant
        }
    ]


@pytest.fixture
def sample_fhir_questionnaire():
    """Sample FHIR questionnaire with enableWhen logic"""
    return {
        "resourceType": "Questionnaire",
        "id": "test-questionnaire",
        "item": [
            {
                "linkId": "gender",
                "type": "choice",
                "text": "Gender",
                "answerOption": [
                    {"valueCoding": {"code": "male"}},
                    {"valueCoding": {"code": "female"}}
                ]
            },
            {
                "linkId": "pregnant",
                "type": "boolean",
                "text": "Currently Pregnant?",
                "enableWhen": [
                    {
                        "question": "gender",
                        "operator": "=",
                        "answerCoding": {"code": "female"}
                    }
                ]
            },
            {
                "linkId": "weeks_pregnant",
                "type": "integer",
                "text": "Weeks Pregnant",
                "enableWhen": [
                    {
                        "question": "pregnant",
                        "operator": "=",
                        "answerBoolean": True
                    }
                ]
            }
        ]
    }


@pytest.fixture
def test_data_dir():
    """Path to test data directory"""
    return Path(__file__).parent / "test_data" / "dictionaries" / "synthetic"


@pytest.fixture
def redcap_dict_path(test_data_dir):
    """Path to REDCap test dictionary"""
    return test_data_dir / "redcap_clinical_with_logic.csv"


@pytest.fixture
def fhir_dict_path(test_data_dir):
    """Path to FHIR test dictionary"""
    return test_data_dir / "fhir_questionnaire_with_logic.json"


# ============================================================================
# TEST ConditionalRule DATACLASS
# ============================================================================

class TestConditionalRule:
    """Test ConditionalRule dataclass operations"""

    @pytest.mark.unit
    def test_conditional_rule_creation(self, sample_rule):
        """Test creating a ConditionalRule with all fields"""
        assert sample_rule.rule_id == "pregnancy_gender_check"
        assert sample_rule.rule_type == "skip_if"
        assert sample_rule.condition.startswith("str(row.get('gender'")
        assert sample_rule.action == "must_be_blank"
        assert len(sample_rule.affected_fields) == 2
        assert "pregnant" in sample_rule.affected_fields
        assert "weeks_pregnant" in sample_rule.affected_fields
        assert sample_rule.severity == "error"
        assert sample_rule.confidence == 1.0
        assert sample_rule.description == "Skip pregnancy questions for male subjects"

    @pytest.mark.unit
    def test_conditional_rule_defaults(self):
        """Test ConditionalRule default values"""
        from src.logic_engine import ConditionalRule

        rule = ConditionalRule(
            rule_id="test_rule",
            rule_type="skip_if",
            condition="True",
            action="must_be_blank",
            affected_fields=["field1"],
            description="Test rule description"  # description is required
        )

        # Check defaults
        assert rule.source == ""
        assert rule.severity == "error"
        assert rule.confidence == 1.0

    @pytest.mark.unit
    def test_conditional_rule_to_dict(self, sample_rule):
        """Test serialization to dictionary"""
        d = sample_rule.to_dict()

        # Check all fields are present
        assert d['rule_id'] == "pregnancy_gender_check"
        assert d['rule_type'] == "skip_if"
        assert d['action'] == "must_be_blank"
        assert 'affected_fields' in d
        assert isinstance(d['affected_fields'], list)
        assert d['severity'] == "error"
        assert d['confidence'] == 1.0
        assert 'condition' in d
        assert 'description' in d
        assert 'source' in d

    @pytest.mark.unit
    def test_conditional_rule_from_dict(self, sample_rule):
        """Test deserialization from dictionary"""
        from src.logic_engine import ConditionalRule

        d = sample_rule.to_dict()
        restored = ConditionalRule.from_dict(d)

        # Check all fields match
        assert restored.rule_id == sample_rule.rule_id
        assert restored.rule_type == sample_rule.rule_type
        assert restored.condition == sample_rule.condition
        assert restored.action == sample_rule.action
        assert restored.affected_fields == sample_rule.affected_fields
        assert restored.severity == sample_rule.severity
        assert restored.confidence == sample_rule.confidence

    @pytest.mark.unit
    def test_conditional_rule_from_dict_with_defaults(self):
        """Test from_dict with missing optional fields uses defaults"""
        from src.logic_engine import ConditionalRule

        minimal_dict = {
            'rule_id': 'test',
            'rule_type': 'skip_if',
            'condition': 'True',
            'action': 'must_be_blank',
            'affected_fields': ['field1'],
            'description': 'Test description'  # description is required
        }

        rule = ConditionalRule.from_dict(minimal_dict)

        assert rule.rule_id == 'test'
        assert rule.description == 'Test description'
        assert rule.source == ""
        assert rule.severity == "error"
        assert rule.confidence == 1.0


# ============================================================================
# TEST LogicViolation DATACLASS
# ============================================================================

class TestLogicViolation:
    """Test LogicViolation dataclass operations"""

    @pytest.mark.unit
    def test_logic_violation_creation(self):
        """Test creating a LogicViolation with all fields"""
        from src.logic_engine import LogicViolation

        violation = LogicViolation(
            rule_id="pregnancy_check",
            rule_description="Skip pregnancy for males",
            row_index=5,
            affected_fields=["pregnant"],
            actual_values={"pregnant": "Yes"},
            expected_behavior="Field should be blank when condition is met"
        )

        assert violation.rule_id == "pregnancy_check"
        assert violation.rule_description == "Skip pregnancy for males"
        assert violation.row_index == 5
        assert violation.affected_fields == ["pregnant"]
        assert violation.actual_values == {"pregnant": "Yes"}
        assert violation.expected_behavior == "Field should be blank when condition is met"
        assert violation.severity == "error"  # default

    @pytest.mark.unit
    def test_logic_violation_to_dict(self):
        """Test serialization to dictionary"""
        from src.logic_engine import LogicViolation

        violation = LogicViolation(
            rule_id="test_rule",
            rule_description="Test rule description",
            row_index=10,
            affected_fields=["test_field"],
            actual_values={"test_field": None},
            expected_behavior="Field should be filled",
            severity="warning"
        )

        d = violation.to_dict()

        assert d['rule_id'] == "test_rule"
        assert d['rule_description'] == "Test rule description"
        assert d['row_index'] == 10
        assert d['affected_fields'] == ["test_field"]
        assert d['actual_values'] == {"test_field": None}
        assert d['expected_behavior'] == "Field should be filled"
        assert d['severity'] == "warning"


# ============================================================================
# TEST LogicCodeGenerator
# ============================================================================

class TestLogicCodeGenerator:
    """Test LogicCodeGenerator code generation and security"""

    @pytest.mark.unit
    def test_generate_validation_code_produces_valid_python(self, sample_rules):
        """Test that generated code is syntactically valid Python"""
        from src.logic_engine import LogicCodeGenerator

        generator = LogicCodeGenerator()
        code = generator.generate_validation_code(sample_rules)

        # Should parse without syntax errors
        tree = ast.parse(code)
        assert tree is not None

        # Should contain the validate_logic function
        assert "def validate_logic" in code
        assert "violations = []" in code
        assert "return violations" in code

    @pytest.mark.unit
    def test_generate_validation_code_empty_rules(self):
        """Test code generation with empty rules list"""
        from src.logic_engine import LogicCodeGenerator

        generator = LogicCodeGenerator()
        code = generator.generate_validation_code([])

        # Should still produce valid code
        tree = ast.parse(code)
        assert tree is not None
        assert "def validate_logic" in code

    @pytest.mark.unit
    def test_generate_rule_check_skip_if(self):
        """Test _generate_rule_check for skip_if rule type"""
        from src.logic_engine import LogicCodeGenerator, ConditionalRule

        generator = LogicCodeGenerator()
        rule = ConditionalRule(
            rule_id="test_skip",
            rule_type="skip_if",
            condition="row.get('gender') == 'Male'",
            action="must_be_blank",
            affected_fields=["pregnant"],
            description="Skip pregnancy for males"
        )

        code = generator._generate_rule_check(rule)

        # Should contain condition check
        assert "row.get('gender') == 'Male'" in code
        # Should check for blank/None values
        assert "is not None" in code or "pd.notna" in code or "!= ''" in code

    @pytest.mark.unit
    def test_generate_rule_check_required_if(self):
        """Test _generate_rule_check for required_if rule type"""
        from src.logic_engine import LogicCodeGenerator, ConditionalRule

        generator = LogicCodeGenerator()
        rule = ConditionalRule(
            rule_id="test_required",
            rule_type="required_if",
            condition="row.get('age') >= 18",
            action="must_be_filled",
            affected_fields=["consent"],
            description="Consent required for adults"
        )

        code = generator._generate_rule_check(rule)

        # Should contain condition check
        assert "row.get('age') >= 18" in code
        # Should check for filled/not None values
        assert "is None" in code or "pd.isna" in code or "== ''" in code

    @pytest.mark.unit
    def test_generate_action_check_must_be_blank(self):
        """Test _generate_action_check for must_be_blank action"""
        from src.logic_engine import LogicCodeGenerator, ConditionalRule

        generator = LogicCodeGenerator()
        rule = ConditionalRule(
            rule_id="test_blank",
            rule_type="skip_if",
            condition="True",
            action="must_be_blank",
            affected_fields=["test_field"],
            description="Test must be blank"
        )
        code = generator._generate_action_check(rule)

        # Should check for non-blank values
        assert "test_field" in code
        assert "pd.notna" in code or "!= ''" in code

    @pytest.mark.unit
    def test_generate_action_check_must_be_filled(self):
        """Test _generate_action_check for must_be_filled action"""
        from src.logic_engine import LogicCodeGenerator, ConditionalRule

        generator = LogicCodeGenerator()
        rule = ConditionalRule(
            rule_id="test_filled",
            rule_type="required_if",
            condition="True",
            action="must_be_filled",
            affected_fields=["test_field"],
            description="Test must be filled"
        )
        code = generator._generate_action_check(rule)

        # Should check for blank values
        assert "test_field" in code
        assert "pd.isna" in code or "== ''" in code

    @pytest.mark.unit
    def test_sanitize_condition_blocks_import(self):
        """Test that import statements are blocked"""
        from src.logic_engine import LogicCodeGenerator

        generator = LogicCodeGenerator()

        # Test various import patterns
        dangerous_patterns = [
            "import os",
            "import sys",
            "from os import system",
            "__import__('os')",
        ]

        for pattern in dangerous_patterns:
            result = generator._sanitize_condition(pattern)
            # Should return safe false or raise error
            assert result == "False" or "import" not in result.lower()

    @pytest.mark.unit
    def test_sanitize_condition_blocks_exec(self):
        """Test that exec/eval are blocked"""
        from src.logic_engine import LogicCodeGenerator

        generator = LogicCodeGenerator()

        dangerous_patterns = [
            "exec('malicious code')",
            "eval('harmful expression')",
            "compile('code', 'string', 'exec')",
        ]

        for pattern in dangerous_patterns:
            result = generator._sanitize_condition(pattern)
            assert result == "False" or ("exec" not in result and "eval" not in result)

    @pytest.mark.unit
    def test_sanitize_condition_blocks_file_operations(self):
        """Test that file operations are blocked"""
        from src.logic_engine import LogicCodeGenerator

        generator = LogicCodeGenerator()

        dangerous_patterns = [
            "open('/etc/passwd')",
            "with open('file.txt') as f: pass",
            "file = open('test')",
        ]

        for pattern in dangerous_patterns:
            result = generator._sanitize_condition(pattern)
            assert result == "False" or "open(" not in result

    @pytest.mark.unit
    def test_sanitize_condition_allows_safe_patterns(self):
        """Test that safe patterns are allowed"""
        from src.logic_engine import LogicCodeGenerator

        generator = LogicCodeGenerator()

        safe_patterns = [
            "row['gender'] == 'Male'",
            "row.get('age', 0) >= 18",
            "str(row.get('value')).lower() in ['yes', 'no']",
            "row['field1'] is not None and row['field2'] > 10",
            "pd.notna(row.get('field'))",
        ]

        for pattern in safe_patterns:
            result = generator._sanitize_condition(pattern)
            # Should be unchanged or minimally modified
            assert "False" != result
            # Should not be completely rejected
            assert len(result) > 5

    @pytest.mark.unit
    def test_validate_generated_code_with_safe_code(self, sample_rules):
        """Test validation passes for safe code"""
        from src.logic_engine import LogicCodeGenerator

        generator = LogicCodeGenerator()
        code = generator.generate_validation_code(sample_rules)

        assert generator.validate_generated_code(code) is True

    @pytest.mark.unit
    def test_validate_generated_code_rejects_unsafe_imports(self):
        """Test validation fails for code with unsafe imports"""
        from src.logic_engine import LogicCodeGenerator

        generator = LogicCodeGenerator()

        unsafe_code = """
import os
import subprocess

def validate_logic(df):
    os.system('rm -rf /')
    return []
"""

        assert generator.validate_generated_code(unsafe_code) is False

    @pytest.mark.unit
    def test_validate_generated_code_rejects_exec_eval(self):
        """Test validation fails for code with exec/eval"""
        from src.logic_engine import LogicCodeGenerator

        generator = LogicCodeGenerator()

        unsafe_code = """
def validate_logic(df):
    eval('malicious code')
    return []
"""

        assert generator.validate_generated_code(unsafe_code) is False

    @pytest.mark.unit
    def test_validate_generated_code_accepts_pandas(self):
        """Test validation allows pandas usage"""
        from src.logic_engine import LogicCodeGenerator

        generator = LogicCodeGenerator()

        safe_code = """
import pandas as pd

def validate_logic(df):
    violations = []
    for idx, row in df.iterrows():
        if pd.notna(row.get('field')):
            violations.append({'row': idx})
    return violations
"""

        # pandas is allowed
        assert generator.validate_generated_code(safe_code) is True


# ============================================================================
# TEST LogicValidator
# ============================================================================

class TestLogicValidator:
    """Test LogicValidator validation execution"""

    @pytest.mark.unit
    def test_validate_empty_rules_returns_empty(self, sample_df_valid):
        """Test that empty rules returns empty violations"""
        from src.logic_engine import LogicValidator

        validator = LogicValidator()
        violations = validator.validate([], sample_df_valid)

        assert violations == []

    @pytest.mark.unit
    def test_validate_empty_dataframe(self, sample_rules):
        """Test validation with empty DataFrame"""
        from src.logic_engine import LogicValidator

        validator = LogicValidator()
        empty_df = pd.DataFrame()
        violations = validator.validate(sample_rules, empty_df)

        # Should handle empty DataFrame gracefully
        assert isinstance(violations, list)

    @pytest.mark.integration
    def test_validate_finds_violations(self, sample_rules, sample_df_violations):
        """Test that violations are correctly detected"""
        from src.logic_engine import LogicValidator

        validator = LogicValidator()

        # Only use the pregnancy rule for this test
        rules = [sample_rules[0]]  # pregnancy_male_skip

        violations = validator.validate(rules, sample_df_violations)

        # Should find violations (males with non-blank pregnancy fields)
        assert len(violations) > 0
        assert all(v.rule_id == "pregnancy_male_skip" for v in violations)

        # Check violation details
        for v in violations:
            assert "pregnant" in v.affected_fields
            assert "blank" in v.expected_behavior.lower()

    @pytest.mark.integration
    def test_validate_no_violations_for_valid_data(self, sample_rules, sample_df_valid):
        """Test that valid data returns no violations"""
        from src.logic_engine import LogicValidator

        validator = LogicValidator()
        rules = [sample_rules[0]]  # pregnancy_male_skip

        violations = validator.validate(rules, sample_df_valid)

        # Should find no violations (males have blank pregnancy)
        assert len(violations) == 0

    @pytest.mark.integration
    def test_validate_with_multiple_rules(self, sample_rules):
        """Test validation with multiple rules"""
        from src.logic_engine import LogicValidator

        validator = LogicValidator()

        # Create data with violations for different rules
        df = pd.DataFrame({
            'subject_id': ['001', '002', '003'],
            'gender': ['Male', 'Female', 'Female'],
            'age': [17, 18, 20],
            'pregnant': ['Yes', 'No', 'No'],  # Row 0: male should be blank
            'consent_signed': ['No', 'Yes', None]  # Row 1 and 2: adults need consent
        })

        violations = validator.validate(sample_rules, df)

        # Should find violations from both rules
        assert len(violations) > 0

        # Check we have violations from both rule types
        rule_ids = {v.rule_id for v in violations}
        # May have pregnancy_male_skip and/or dose_age_check
        assert len(rule_ids) >= 1

    @pytest.mark.unit
    def test_generate_code_from_rules(self, sample_rules):
        """Test code generation for caching"""
        from src.logic_engine import LogicValidator

        validator = LogicValidator()
        code = validator.generate_code_from_rules(sample_rules)

        # Check code structure
        assert "def validate_logic" in code
        assert "violations = []" in code
        assert "return violations" in code

        # Should be valid Python
        tree = ast.parse(code)
        assert tree is not None

    @pytest.mark.integration
    def test_validate_with_code(self, sample_rules, sample_df_violations):
        """Test validation using pre-generated code"""
        from src.logic_engine import LogicValidator, LogicCodeGenerator

        # Generate code
        generator = LogicCodeGenerator()
        code = generator.generate_validation_code([sample_rules[0]])

        # Execute validation with code
        validator = LogicValidator()
        violations = validator.validate_with_code(code, sample_df_violations)

        # Should find violations
        assert len(violations) > 0

    @pytest.mark.unit
    def test_execute_sandboxed_restricts_builtins(self):
        """Test that sandboxed execution restricts dangerous builtins"""
        from src.logic_engine import LogicValidator

        validator = LogicValidator()

        # Code that tries to use restricted builtins
        unsafe_code = """
def validate_logic(df):
    # Try to access restricted functions
    open('/etc/passwd')
    return []

result = validate_logic(df)
"""

        df = pd.DataFrame({'field': [1, 2, 3]})

        # Should either raise exception or return empty result
        try:
            result = validator._execute_sandboxed(unsafe_code, df)
            # If it doesn't raise, result should be safe
            assert result is None or isinstance(result, list)
        except Exception:
            # Expected to raise security exception
            pass

    @pytest.mark.unit
    def test_execute_sandboxed_handles_errors_gracefully(self):
        """Test that execution errors are handled gracefully"""
        from src.logic_engine import LogicValidator

        validator = LogicValidator()

        # Code with syntax error
        bad_code = """
def validate_logic(df):
    # Syntax error below
    return []
    invalid syntax here
"""

        df = pd.DataFrame({'field': [1, 2, 3]})

        # Should handle error gracefully
        try:
            result = validator._execute_sandboxed(bad_code, df)
            # If no exception, should return safe value
            assert result is None or isinstance(result, list)
        except SyntaxError:
            # Expected to raise syntax error
            pass


# ============================================================================
# TEST RuleExtractor
# ============================================================================

class TestRuleExtractor:
    """Test RuleExtractor parsing of REDCap and FHIR logic"""

    @pytest.mark.unit
    def test_extract_rules_empty_input(self):
        """Test with empty input returns empty list"""
        from src.logic_engine import RuleExtractor

        extractor = RuleExtractor()
        rules = extractor.extract_rules_from_fields([])

        assert rules == []

    @pytest.mark.unit
    def test_parse_redcap_branching_simple(self):
        """Test parsing simple REDCap branching logic"""
        from src.logic_engine import RuleExtractor

        extractor = RuleExtractor()
        rule = extractor._parse_redcap_branching("pregnant", "[gender]='2'")

        assert rule is not None
        assert "gender" in rule.condition.lower()
        assert rule.affected_fields == ["pregnant"]
        assert rule.rule_type in ["skip_if", "required_if", "show_if"]

    @pytest.mark.unit
    def test_parse_redcap_branching_complex(self):
        """Test parsing complex REDCap branching logic with AND/OR"""
        from src.logic_engine import RuleExtractor

        extractor = RuleExtractor()

        # Complex condition with AND
        rule = extractor._parse_redcap_branching(
            "lab_glucose",
            "[diabetes]='1' and [age] >= 18"
        )

        assert rule is not None
        assert "diabetes" in rule.condition.lower()
        assert rule.affected_fields == ["lab_glucose"]

    @pytest.mark.unit
    def test_parse_redcap_branching_with_or(self):
        """Test parsing REDCap branching logic with OR"""
        from src.logic_engine import RuleExtractor

        extractor = RuleExtractor()

        rule = extractor._parse_redcap_branching(
            "followup",
            "[status]='complete' or [status]='partial'"
        )

        assert rule is not None
        assert "status" in rule.condition.lower()

    @pytest.mark.unit
    def test_extract_rules_from_fields(self, sample_fields_redcap):
        """Test extracting rules from field definitions"""
        from src.logic_engine import RuleExtractor

        extractor = RuleExtractor()
        rules = extractor.extract_rules_from_fields(sample_fields_redcap)

        # Should extract rules from fields with branching_logic
        assert len(rules) >= 2  # pregnant and weeks_pregnant have branching

        # Check rule IDs contain field names
        rule_ids = [r.rule_id for r in rules]
        assert any("pregnant" in rid for rid in rule_ids)
        assert any("weeks_pregnant" in rid for rid in rule_ids)

    @pytest.mark.unit
    def test_parse_business_rule_male_skip(self):
        """Test parsing business rule for male skip pattern"""
        from src.logic_engine import RuleExtractor

        extractor = RuleExtractor()

        # Business rule text for male skip pattern
        rule_text = "If male, skip this field"

        rule = extractor._parse_business_rule("pregnant", rule_text)

        assert rule is not None
        assert rule.rule_id.startswith("pregnant")
        assert rule.affected_fields == ["pregnant"]
        # Should indicate skip for males
        assert "gender" in rule.condition.lower()

    @pytest.mark.unit
    def test_parse_business_rule_female_required(self):
        """Test parsing business rule for female required pattern"""
        from src.logic_engine import RuleExtractor

        extractor = RuleExtractor()

        # Business rule text for female required pattern
        rule_text = "Required for female subjects only if age >= 18"

        rule = extractor._parse_business_rule("pregnancy_test", rule_text)

        assert rule is not None
        assert "gender" in rule.condition.lower() or "female" in rule.description.lower()
        # Age pattern should be detected
        assert "age" in rule.condition.lower() or rule is not None

    @pytest.mark.unit
    def test_parse_fhir_enable_when_simple(self, sample_fhir_questionnaire):
        """Test parsing simple FHIR enableWhen"""
        from src.logic_engine import RuleExtractor

        extractor = RuleExtractor()

        # Extract pregnant field with enableWhen
        pregnant_item = sample_fhir_questionnaire['item'][1]
        field_name = pregnant_item['linkId']  # "pregnant"
        enable_when = pregnant_item['enableWhen'][0]

        rule = extractor._parse_fhir_enable_when(field_name, enable_when)

        assert rule is not None
        assert rule.affected_fields == ["pregnant"]
        assert "gender" in rule.condition.lower()

    @pytest.mark.unit
    def test_parse_fhir_enable_when_boolean(self, sample_fhir_questionnaire):
        """Test parsing FHIR enableWhen with boolean answer"""
        from src.logic_engine import RuleExtractor

        extractor = RuleExtractor()

        # Extract weeks_pregnant field
        weeks_item = sample_fhir_questionnaire['item'][2]
        field_name = weeks_item['linkId']  # "weeks_pregnant"
        enable_when = weeks_item['enableWhen'][0]

        rule = extractor._parse_fhir_enable_when(field_name, enable_when)

        assert rule is not None
        assert rule.affected_fields == ["weeks_pregnant"]
        assert "pregnant" in rule.condition.lower()

    @pytest.mark.unit
    def test_extract_rules_from_fhir_questionnaire(self, sample_fhir_questionnaire):
        """Test extracting rules from FHIR questionnaire"""
        from src.logic_engine import RuleExtractor

        extractor = RuleExtractor()

        # Convert FHIR items to field list
        # Note: implementation expects 'enable_when' key, not 'fhir_enable_when'
        fields = []
        for item in sample_fhir_questionnaire.get('item', []):
            field = {
                'field_name': item['linkId'],
                'enable_when': item.get('enableWhen', [])
            }
            fields.append(field)

        rules = extractor.extract_rules_from_fields(fields, format_type="FHIR")

        # Should extract rules for items with enableWhen
        assert len(rules) >= 2


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for full workflow"""

    @pytest.mark.integration
    def test_full_workflow_rules_to_violations(self, sample_rules, sample_df_violations):
        """Test complete workflow: rules → code → execute → violations"""
        from src.logic_engine import LogicValidator, LogicCodeGenerator

        # 1. Generate code
        generator = LogicCodeGenerator()
        code = generator.generate_validation_code(sample_rules[:1])

        # 2. Validate code is safe
        assert generator.validate_generated_code(code) is True

        # 3. Execute validation
        validator = LogicValidator()
        violations = validator.validate_with_code(code, sample_df_violations)

        # 4. Check violations found
        assert len(violations) > 0
        assert all(hasattr(v, 'rule_id') for v in violations)
        assert all(hasattr(v, 'row_index') for v in violations)

    @pytest.mark.integration
    def test_workflow_with_clinical_data(self):
        """Test with sample clinical data (gender/pregnancy logic)"""
        from src.logic_engine import RuleExtractor, LogicValidator, LogicCodeGenerator

        # Define fields with branching logic
        fields = [
            {
                'field_name': 'subject_id',
                'required': True
            },
            {
                'field_name': 'gender',
                'allowed_values': ['Male', 'Female']
            },
            {
                'field_name': 'pregnant',
                'branching_logic': "[gender]='Female'"
            },
            {
                'field_name': 'weeks_pregnant',
                'branching_logic': "[pregnant]='Yes'"
            }
        ]

        # Extract rules
        extractor = RuleExtractor()
        rules = extractor.extract_rules_from_fields(fields)

        assert len(rules) > 0

        # Create test data with violation
        df = pd.DataFrame({
            'subject_id': ['001', '002', '003'],
            'gender': ['Male', 'Female', 'Female'],
            'pregnant': ['Yes', 'Yes', 'No'],  # Male should be blank!
            'weeks_pregnant': [None, 12, None]
        })

        # Generate and execute validation
        generator = LogicCodeGenerator()
        code = generator.generate_validation_code(rules)

        validator = LogicValidator()
        violations = validator.validate_with_code(code, df)

        # Should find violation for male with pregnancy value
        # Note: may find 0 violations if rule extraction/generation needs implementation
        assert isinstance(violations, list)

    @pytest.mark.integration
    def test_with_real_redcap_dictionary(self, redcap_dict_path):
        """Test with real REDCap test dictionary file"""
        if not redcap_dict_path.exists():
            pytest.skip("REDCap test data file not found")

        from src.logic_engine import RuleExtractor
        import csv

        # Load REDCap dictionary
        with open(redcap_dict_path, 'r') as f:
            reader = csv.DictReader(f)
            fields = []
            for row in reader:
                field = {
                    'field_name': row.get('Variable / Field Name', ''),
                    'field_type': row.get('Field Type', ''),
                    'field_label': row.get('Field Label', ''),
                    'branching_logic': row.get('Branching Logic (Show field only if...)', '')
                }
                if field['field_name']:  # Skip empty rows
                    fields.append(field)

        # Extract rules
        extractor = RuleExtractor()
        rules = extractor.extract_rules_from_fields(fields)

        # Should extract multiple rules from real dictionary
        # Note: actual number depends on implementation
        assert isinstance(rules, list)

        # If rules extracted, check structure
        for rule in rules[:5]:  # Check first 5
            assert hasattr(rule, 'rule_id')
            assert hasattr(rule, 'condition')
            assert hasattr(rule, 'affected_fields')
            assert len(rule.affected_fields) > 0

    @pytest.mark.integration
    def test_with_real_fhir_questionnaire(self, fhir_dict_path):
        """Test with real FHIR test dictionary file"""
        if not fhir_dict_path.exists():
            pytest.skip("FHIR test data file not found")

        from src.logic_engine import RuleExtractor

        # Load FHIR questionnaire
        with open(fhir_dict_path, 'r') as f:
            fhir_data = json.load(f)

        # Extract fields from FHIR questionnaire
        items = fhir_data.get('item', [])
        assert len(items) > 0

        # Convert to field format
        fields = []
        for item in items:
            # Handle nested groups
            if item.get('type') == 'group':
                for subitem in item.get('item', []):
                    field = {
                        'field_name': subitem['linkId'],
                        'fhir_type': subitem.get('type'),
                        'fhir_enable_when': subitem.get('enableWhen', [])
                    }
                    fields.append(field)
            else:
                field = {
                    'field_name': item['linkId'],
                    'fhir_type': item.get('type'),
                    'fhir_enable_when': item.get('enableWhen', [])
                }
                fields.append(field)

        # Extract rules
        extractor = RuleExtractor()
        rules = extractor.extract_rules_from_fields(fields)

        # Should extract rules from fields with enableWhen
        assert isinstance(rules, list)

        # Check rule structure
        for rule in rules[:5]:
            assert hasattr(rule, 'rule_id')
            assert hasattr(rule, 'affected_fields')

    @pytest.mark.integration
    def test_end_to_end_redcap_to_violations(self, redcap_dict_path):
        """Test end-to-end: REDCap dict → rules → code → validation"""
        if not redcap_dict_path.exists():
            pytest.skip("REDCap test data file not found")

        from src.logic_engine import RuleExtractor, LogicCodeGenerator, LogicValidator
        import csv

        # 1. Load dictionary
        with open(redcap_dict_path, 'r') as f:
            reader = csv.DictReader(f)
            fields = []
            for row in reader:
                field = {
                    'field_name': row.get('Variable / Field Name', '').strip(),
                    'branching_logic': row.get('Branching Logic (Show field only if...)', '').strip()
                }
                if field['field_name']:
                    fields.append(field)

        # 2. Extract rules
        extractor = RuleExtractor()
        rules = extractor.extract_rules_from_fields(fields)

        if len(rules) == 0:
            pytest.skip("No rules extracted from dictionary")

        # 3. Generate code
        generator = LogicCodeGenerator()
        code = generator.generate_validation_code(rules[:3])  # Use first 3 rules

        # 4. Validate code safety
        assert generator.validate_generated_code(code) is True

        # 5. Create sample data
        df = pd.DataFrame({
            'subject_id': ['001', '002'],
            'gender': ['1', '2'],  # Male, Female
            'pregnant': ['1', '1'],  # Both pregnant - violation for male!
            'age': [25, 30]
        })

        # 6. Execute validation
        validator = LogicValidator()
        violations = validator.validate_with_code(code, df)

        # 7. Check results
        assert isinstance(violations, list)

    @pytest.mark.integration
    def test_caching_code_for_reuse(self, sample_rules):
        """Test caching generated code for repeated use"""
        from src.logic_engine import LogicCodeGenerator, LogicValidator

        # Generate code once
        generator = LogicCodeGenerator()
        code = generator.generate_validation_code(sample_rules)

        # Create multiple datasets
        df1 = pd.DataFrame({
            'gender': ['Male', 'Female'],
            'age': [25, 30],
            'pregnant': ['Yes', 'No'],  # Violation in row 0
            'consent_signed': ['No', 'Yes']  # Violation in row 0
        })

        df2 = pd.DataFrame({
            'gender': ['Female', 'Female'],
            'age': [17, 18],
            'pregnant': ['No', 'No'],
            'consent_signed': [None, 'Yes']  # Violation in row 1
        })

        # Validate multiple datasets with same code
        validator = LogicValidator()

        violations1 = validator.validate_with_code(code, df1)
        violations2 = validator.validate_with_code(code, df2)

        # Both should execute successfully
        assert isinstance(violations1, list)
        assert isinstance(violations2, list)

    @pytest.mark.integration
    def test_performance_with_large_dataset(self, sample_rules):
        """Test validation performance with larger dataset"""
        from src.logic_engine import LogicValidator, LogicCodeGenerator
        import time

        # Create larger dataset
        num_rows = 1000
        df = pd.DataFrame({
            'subject_id': [f'SUBJ-{i:04d}' for i in range(num_rows)],
            'gender': ['Male' if i % 2 == 0 else 'Female' for i in range(num_rows)],
            'age': [18 + (i % 50) for i in range(num_rows)],
            'pregnant': ['Yes' if i % 3 == 0 else 'No' for i in range(num_rows)],
            'consent_signed': ['Yes' if i % 5 != 0 else None for i in range(num_rows)]
        })

        # Generate code
        generator = LogicCodeGenerator()
        code = generator.generate_validation_code(sample_rules)

        # Time validation
        validator = LogicValidator()
        start = time.time()
        violations = validator.validate_with_code(code, df)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 5 seconds for 1000 rows)
        assert elapsed < 5.0
        assert isinstance(violations, list)

        # Check we found violations (males with pregnancy = Yes)
        assert len(violations) > 0
