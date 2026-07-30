"""
Logic Validation Engine for Data Analyzer

This module provides conditional logic validation that EXTENDS the existing QualityChecker.
- QualityChecker: Validates individual fields (types, ranges, allowed values)
- LogicValidator: Validates conditional logic ACROSS fields

Key Components:
- Condition (and subclasses): a structured, data-only representation of a rule
  predicate (NOT executable code). Compare/And/Or/Not/Const form a small tree.
- ConditionalRule: Represents a conditional logic rule from a data dictionary.
- LogicViolation: Represents a violation of a conditional logic rule.
- LogicValidator: Evaluates rules against a DataFrame using vectorized pandas ops.
- RuleExtractor: Extracts ConditionalRules from parsed dictionary fields.

Security model (rewritten — "Option B"):
- Conditions are STRUCTURED DATA, never source code. There is NO code generation,
  NO ``exec``/``eval``/``compile``, and NO string-to-Python conversion anywhere in
  the evaluation path. A malicious data dictionary cannot inject executable code
  because no dictionary-derived text is ever executed.
- Every predicate is one of a fixed, closed set of typed nodes (Compare/And/Or/
  Not/Const), each evaluated by hand-written, vectorized pandas operations.
- Anything that cannot be parsed into that closed set becomes ``Const(False)``
  (the rule simply does not fire) — fail-closed, never fail-open into execution.

This replaces the previous generate-Python-then-``exec`` design, which relied on a
regex denylist + AST gate that were bypassable (e.g. via ``pd.read_pickle``).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np
import json
import re
import logging

logger = logging.getLogger(__name__)

# Guardrails against pathological dictionary input (DoS), not against injection —
# injection is structurally impossible now.
_MAX_EXPRESSION_LEN = 2000
_MAX_PARSE_DEPTH = 50


# =============================================================================
# STRUCTURED CONDITION TREE  (data, not code)
# =============================================================================

# Comparison operators understood by the evaluator. This is the COMPLETE set —
# there is no escape hatch to arbitrary expressions.
_STRING_OPS = {"eq", "ne", "in", "not_in", "contains"}
_NUMERIC_OPS = {"gt", "lt", "ge", "le"}
_BLANK_OPS = {"is_blank", "not_blank"}
VALID_OPS = _STRING_OPS | _NUMERIC_OPS | _BLANK_OPS


def _all_false(df: pd.DataFrame) -> pd.Series:
    return pd.Series(False, index=df.index)


def _all_true(df: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=df.index)


def _norm_scalar(value: Any) -> str:
    """Normalize a scalar for case-insensitive string comparison."""
    return str(value).strip().lower()


def _str_norm(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _is_blank_series(series: pd.Series) -> pd.Series:
    """True where a cell is NaN/None or an empty/whitespace string."""
    return series.isna() | (series.astype(str).str.strip() == "")


class Condition:
    """
    Base class for the structured condition tree.

    A Condition is pure data. ``evaluate(df)`` returns a boolean pandas Series
    aligned to ``df.index`` (True where the predicate holds for that row). No node
    ever executes strings, imports modules, or touches the filesystem/network.
    """

    def evaluate(self, df: pd.DataFrame) -> pd.Series:  # pragma: no cover - abstract
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    @staticmethod
    def from_dict(data: Any) -> "Condition":
        """
        Rebuild a Condition from its dict form.

        Fail-closed: anything unrecognized (including a legacy *string* condition
        from an older cache) becomes Const(False) — it is NEVER interpreted as
        code.
        """
        if isinstance(data, Condition):
            return data
        if not isinstance(data, dict):
            if isinstance(data, str):
                # Legacy cached rules stored a Python expression string here.
                # We deliberately do NOT execute it; the rule is neutralized.
                logger.warning(
                    "Encountered legacy string condition; neutralizing to Const(False): %r",
                    data[:120],
                )
            return Const(False)

        ctype = data.get("type")
        try:
            if ctype == "compare":
                return Compare(data["field"], data["op"], data.get("value"))
            if ctype == "and":
                return And([Condition.from_dict(o) for o in data.get("operands", [])])
            if ctype == "or":
                return Or([Condition.from_dict(o) for o in data.get("operands", [])])
            if ctype == "not":
                return Not(Condition.from_dict(data.get("operand")))
            if ctype == "const":
                return Const(bool(data.get("value", False)))
        except Exception as e:  # malformed node -> fail closed
            logger.warning("Malformed condition node %r: %s", data, e)
        return Const(False)


@dataclass
class Const(Condition):
    """A constant boolean predicate (used for unparseable / always-true cases)."""
    value: bool = False

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        return _all_true(df) if self.value else _all_false(df)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "const", "value": bool(self.value)}

    def __str__(self) -> str:
        return "True" if self.value else "False"


@dataclass
class Compare(Condition):
    """
    A single field comparison.

    Semantics by op:
    - eq/ne         : case-insensitive string comparison (e.g. 2 == '2')
    - in/not_in     : case-insensitive membership; ``value`` is a list
    - contains      : case-insensitive substring test
    - gt/lt/ge/le   : numeric comparison (non-numeric cells -> not met)
    - is_blank/not_blank : NaN or empty/whitespace string (``value`` ignored)

    A missing column always evaluates to False (the condition is "not met"),
    which prevents spurious violations.
    """
    field: str
    op: str
    value: Any = None

    def __post_init__(self):
        if self.op not in VALID_OPS:
            raise ValueError(f"Unsupported comparison op: {self.op!r}")

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        if self.field not in df.columns:
            return _all_false(df)
        col = df[self.field]

        if self.op == "is_blank":
            return _is_blank_series(col)
        if self.op == "not_blank":
            return ~_is_blank_series(col)

        if self.op in _NUMERIC_OPS:
            n = pd.to_numeric(col, errors="coerce")
            try:
                thr = float(self.value)
            except (TypeError, ValueError):
                return _all_false(df)
            if self.op == "gt":
                res = n > thr
            elif self.op == "ge":
                res = n >= thr
            elif self.op == "lt":
                res = n < thr
            else:  # le
                res = n <= thr
            return (res & n.notna()).fillna(False)

        # String / membership ops
        s = _str_norm(col)
        if self.op == "eq":
            return s == _norm_scalar(self.value)
        if self.op == "ne":
            return s != _norm_scalar(self.value)
        if self.op == "in":
            values = self.value if isinstance(self.value, (list, tuple, set)) else [self.value]
            return s.isin([_norm_scalar(v) for v in values])
        if self.op == "not_in":
            values = self.value if isinstance(self.value, (list, tuple, set)) else [self.value]
            return ~s.isin([_norm_scalar(v) for v in values])
        if self.op == "contains":
            return s.str.contains(_norm_scalar(self.value), regex=False, na=False)

        return _all_false(df)  # unreachable given __post_init__ guard

    def to_dict(self) -> Dict[str, Any]:
        value = list(self.value) if isinstance(self.value, (set, tuple)) else self.value
        return {"type": "compare", "field": self.field, "op": self.op, "value": value}

    def __str__(self) -> str:
        return f"{self.field} {self.op} {self.value!r}"


@dataclass
class And(Condition):
    operands: List[Condition] = field(default_factory=list)

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        res = _all_true(df)
        for op in self.operands:
            res &= op.evaluate(df)
        return res

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "and", "operands": [o.to_dict() for o in self.operands]}

    def __str__(self) -> str:
        return "(" + " and ".join(str(o) for o in self.operands) + ")"


@dataclass
class Or(Condition):
    operands: List[Condition] = field(default_factory=list)

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        res = _all_false(df)
        for op in self.operands:
            res |= op.evaluate(df)
        return res

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "or", "operands": [o.to_dict() for o in self.operands]}

    def __str__(self) -> str:
        return "(" + " or ".join(str(o) for o in self.operands) + ")"


@dataclass
class Not(Condition):
    operand: Condition = field(default_factory=lambda: Const(True))

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        return ~self.operand.evaluate(df)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "not", "operand": self.operand.to_dict()}

    def __str__(self) -> str:
        return f"not {self.operand}"


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
    - value_in: Field value must be in allowed set (not yet implemented)

    Attributes:
        rule_id: Unique identifier for the rule
        rule_type: Type of conditional rule
        condition: STRUCTURED Condition tree (data, not code) that decides when
            the rule applies. Legacy string conditions are neutralized on load.
        action: Action to take when condition is met
        affected_fields: List of field names this rule applies to
        description: Human-readable description of the rule
        source: Original dictionary text (for traceability)
        severity: "error" or "warning"
        confidence: 0.0-1.0 (1.0 = explicit in dictionary, <1.0 = inferred)
    """
    rule_id: str
    rule_type: str
    condition: Condition
    action: str
    affected_fields: List[str]
    description: str
    source: str = ""
    severity: str = "error"
    confidence: float = 1.0

    def __post_init__(self):
        # Accept dict/str forms defensively (e.g. deserialization, legacy data)
        # and coerce to a real Condition. A bare string is NEVER executed.
        if not isinstance(self.condition, Condition):
            self.condition = Condition.from_dict(self.condition)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type,
            "condition": self.condition.to_dict(),
            "action": self.action,
            "affected_fields": list(self.affected_fields),
            "description": self.description,
            "source": self.source,
            "severity": self.severity,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConditionalRule":
        """Create ConditionalRule from dictionary (condition rebuilt fail-closed)."""
        allowed = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        allowed["condition"] = Condition.from_dict(allowed.get("condition"))
        return cls(**allowed)


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
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# =============================================================================
# LOGIC VALIDATOR
# =============================================================================

def _to_py(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python and NaN/NA to None (JSON-safe)."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


class LogicValidator:
    """
    Validates conditional logic rules against data.

    This class EXTENDS the existing QualityChecker - it does NOT replace it.
    - QualityChecker: Handles field types, value ranges, allowed values
    - LogicValidator: Handles conditional logic BETWEEN fields

    Security:
    - Rules are evaluated as structured data via vectorized pandas operations.
    - No code generation, no exec/eval, no filesystem/network access.
    """

    def __init__(self):
        """Initialize LogicValidator."""
        pass

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

        violations: List[LogicViolation] = []
        for rule in rules:
            try:
                violations.extend(self._check_rule(rule, df))
            except Exception as e:
                # A single bad rule must never abort validation of the rest.
                logger.warning(f"Skipping rule {rule.rule_id} due to error: {e}")

        logger.info(f"Found {len(violations)} logic violations")
        return violations

    def _check_rule(self, rule: ConditionalRule, df: pd.DataFrame) -> List[LogicViolation]:
        """Evaluate one rule and collect its violations (vectorized)."""
        action = (rule.action or "").lower()

        if action in ("must_be_blank", "skip"):
            expect_blank = True
            expected_behavior = "Field should be blank when condition is met"
        elif action in ("must_be_filled", "required"):
            expect_blank = False
            expected_behavior = "Field should be filled when condition is met"
        else:
            # value_in / unknown actions are not evaluated (parity with prior code).
            if action not in ("value_in",):
                logger.debug(f"Unknown action '{action}' for rule {rule.rule_id}; skipping")
            return []

        condition_met = rule.condition.evaluate(df)  # boolean Series

        present_fields = [f for f in rule.affected_fields if f in df.columns]
        results: List[LogicViolation] = []

        for fname in present_fields:
            blank = _is_blank_series(df[fname])
            # violation when condition is met AND the field's fill-state is wrong
            offending = condition_met & (blank if not expect_blank else ~blank)
            if not offending.any():
                continue

            for idx in df.index[offending]:
                actual_values = {
                    f: _to_py(df.at[idx, f])
                    for f in rule.affected_fields
                    if f in df.columns
                }
                results.append(
                    LogicViolation(
                        rule_id=rule.rule_id,
                        rule_description=rule.description,
                        row_index=self._row_number(df, idx),
                        affected_fields=list(rule.affected_fields),
                        actual_values=actual_values,
                        expected_behavior=expected_behavior,
                        severity=rule.severity,
                    )
                )
        return results

    @staticmethod
    def _row_number(df: pd.DataFrame, idx: Any) -> int:
        """Best-effort 0-indexed row number for a violation."""
        try:
            return int(idx)
        except (TypeError, ValueError):
            try:
                return int(df.index.get_loc(idx))
            except Exception:
                return -1


# =============================================================================
# REDCap EXPRESSION PARSER  (text -> structured Condition, never code)
# =============================================================================

class _ParseError(Exception):
    pass


# Tokenizer for the small REDCap branching-logic grammar.
_TOKEN_RE = re.compile(
    r"""
    (?P<WS>\s+)
  | (?P<FIELD>\[[A-Za-z0-9_]+(?:\([A-Za-z0-9_]+\))?\])
  | (?P<NUMBER>-?\d+(?:\.\d+)?)
  | (?P<STRING>'[^']*'|"[^"]*")
  | (?P<OP><>|!=|>=|<=|=|>|<)
  | (?P<AND>\band\b)
  | (?P<OR>\bor\b)
  | (?P<LPAREN>\()
  | (?P<RPAREN>\))
    """,
    re.VERBOSE | re.IGNORECASE,
)

_OP_MAP = {"=": "eq", "==": "eq", "<>": "ne", "!=": "ne", ">": "gt", "<": "lt", ">=": "ge", "<=": "le"}


def _field_from_token(tok: str) -> str:
    """[gender] -> gender ; [chk(2)] -> chk___2  (REDCap checkbox export naming)."""
    inner = tok[1:-1]
    m = re.match(r"^([A-Za-z0-9_]+)\(([A-Za-z0-9_]+)\)$", inner)
    if m:
        return f"{m.group(1)}___{m.group(2)}"
    return inner


def _tokenize(text: str):
    tokens = []
    pos = 0
    n = len(text)
    while pos < n:
        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise _ParseError(f"Unexpected character at {pos}: {text[pos:pos+10]!r}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "WS":
            continue
        tokens.append((kind, m.group()))
    return tokens


def parse_redcap_expression(text: str) -> Condition:
    """
    Parse a REDCap branching-logic string into a structured Condition.

    Grammar (subset):
        expr    := or_expr
        or_expr := and_expr ("or" and_expr)*
        and_expr:= atom ("and" atom)*
        atom    := "(" expr ")" | comparison
        comparison := FIELD OP (NUMBER | STRING)

    Fail-closed: any parse problem returns Const(False) so the rule simply does
    not fire. No part of ``text`` is ever executed.
    """
    if not text or not text.strip():
        return Const(False)
    if len(text) > _MAX_EXPRESSION_LEN:
        logger.warning("REDCap expression too long (%d chars); neutralizing", len(text))
        return Const(False)

    try:
        tokens = _tokenize(text)
        parser = _RedcapParser(tokens)
        cond = parser.parse()
        parser.expect_end()
        return cond
    except _ParseError as e:
        logger.warning("Could not parse REDCap expression %r: %s", text, e)
        return Const(False)


class _RedcapParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.i = 0

    def _peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else (None, None)

    def _advance(self):
        tok = self._peek()
        self.i += 1
        return tok

    def expect_end(self):
        if self.i != len(self.tokens):
            raise _ParseError(f"Trailing tokens from {self._peek()}")

    def parse(self, depth: int = 0) -> Condition:
        if depth > _MAX_PARSE_DEPTH:
            raise _ParseError("Expression nested too deeply")
        return self._parse_or(depth)

    def _parse_or(self, depth: int) -> Condition:
        node = self._parse_and(depth)
        operands = [node]
        while self._peek()[0] == "OR":
            self._advance()
            operands.append(self._parse_and(depth))
        return operands[0] if len(operands) == 1 else Or(operands)

    def _parse_and(self, depth: int) -> Condition:
        node = self._parse_atom(depth)
        operands = [node]
        while self._peek()[0] == "AND":
            self._advance()
            operands.append(self._parse_atom(depth))
        return operands[0] if len(operands) == 1 else And(operands)

    def _parse_atom(self, depth: int) -> Condition:
        kind, value = self._peek()
        if kind == "LPAREN":
            self._advance()
            node = self.parse(depth + 1)
            if self._peek()[0] != "RPAREN":
                raise _ParseError("Missing closing parenthesis")
            self._advance()
            return node
        return self._parse_comparison()

    def _parse_comparison(self) -> Condition:
        kind, value = self._advance()
        if kind != "FIELD":
            raise _ParseError(f"Expected field, got {kind} {value!r}")
        field_name = _field_from_token(value)

        op_kind, op_val = self._advance()
        if op_kind != "OP":
            raise _ParseError(f"Expected operator, got {op_kind} {op_val!r}")
        op = _OP_MAP.get(op_val)
        if op is None:
            raise _ParseError(f"Unknown operator {op_val!r}")

        val_kind, val_raw = self._advance()
        if val_kind == "STRING":
            operand: Any = val_raw[1:-1]
        elif val_kind == "NUMBER":
            operand = float(val_raw) if "." in val_raw else int(val_raw)
        else:
            raise _ParseError(f"Expected value, got {val_kind} {val_raw!r}")

        # Relational operators require numeric comparison; '='/'<>' use string eq/ne.
        if op in _NUMERIC_OPS:
            try:
                operand = float(operand)
            except (TypeError, ValueError):
                raise _ParseError(f"Non-numeric value for {op}: {operand!r}")
        return Compare(field_name, op, operand)


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
    ConditionalRule objects (with STRUCTURED conditions) that LogicValidator
    can evaluate. No dictionary text is ever turned into executable code.
    """

    def extract_rules_from_fields(self,
                                  fields: List[Dict],
                                  format_type: str = "REDCap") -> List[ConditionalRule]:
        """
        Extract conditional rules from field definitions.

        Priority order:
        1. LLM-extracted conditional_rules (works with ANY format)
        2. Format-specific parsers (REDCap/FHIR) as fallback
        3. Business rules text parsing

        Args:
            fields: List of field dictionaries from LLM parsing
            format_type: "REDCap", "FHIR", "Custom", etc.

        Returns:
            List of ConditionalRule objects
        """
        rules: List[ConditionalRule] = []

        for field_def in fields:
            if not isinstance(field_def, dict):
                continue
            field_name = field_def.get('field_name', '')
            if not field_name:
                continue

            # Isolate per-field failures: a single malformed field (e.g. a
            # non-string branching_logic) must never abort extraction for the
            # rest of the dictionary and silently disable logic validation.
            try:
                # PRIORITY 1: LLM-extracted conditional_rules (any format)
                llm_rules = self._extract_llm_rules(field_def)
                if llm_rules:
                    rules.extend(llm_rules)
                    logger.debug(f"Field '{field_name}': Using {len(llm_rules)} LLM-extracted rule(s)")

                # PRIORITY 2: Format-specific parsers as fallback/complement
                if format_type.lower() == "redcap":
                    rules.extend(self._extract_redcap_rules(field_def))
                elif format_type.lower() == "fhir":
                    rules.extend(self._extract_fhir_rules(field_def))
                else:
                    rules.extend(self._extract_custom_rules(field_def))
            except Exception as e:
                logger.warning(f"Failed to extract rules for field '{field_name}': {e}")
                continue

        logger.info(f"Extracted {len(rules)} rules from {len(fields)} fields")
        return rules

    def _extract_redcap_rules(self, field_def: Dict) -> List[ConditionalRule]:
        """Extract rules from a REDCap field definition."""
        rules: List[ConditionalRule] = []
        field_name = field_def.get('field_name', '')

        branching_logic = field_def.get('branching_logic', '')
        if branching_logic:
            rule = self._parse_redcap_branching(field_name, branching_logic)
            if rule:
                rules.append(rule)

        business_rules = field_def.get('business_rules', [])
        if isinstance(business_rules, list):
            for br in business_rules:
                rule = self._parse_business_rule(field_name, br)
                if rule:
                    rules.append(rule)

        return rules

    def _extract_fhir_rules(self, field_def: Dict) -> List[ConditionalRule]:
        """Extract rules from a FHIR questionnaire item."""
        rules: List[ConditionalRule] = []
        field_name = field_def.get('field_name', '')

        enable_when = field_def.get('enable_when', [])
        if isinstance(enable_when, list):
            for ew in enable_when:
                rule = self._parse_fhir_enable_when(field_name, ew)
                if rule:
                    rules.append(rule)

        return rules

    def _extract_custom_rules(self, field_def: Dict) -> List[ConditionalRule]:
        """Extract rules from a custom field definition."""
        rules: List[ConditionalRule] = []
        field_name = field_def.get('field_name', '')

        business_rules = field_def.get('business_rules', [])
        if isinstance(business_rules, list):
            for br in business_rules:
                rule = self._parse_business_rule(field_name, br)
                if rule:
                    rules.append(rule)

        return rules

    def _parse_redcap_branching(self, field_name: str, branching_logic: str) -> Optional[ConditionalRule]:
        """
        Parse REDCap branching logic into a ConditionalRule.

        REDCap syntax examples:
        - [gender]='2'                 -> Show field if gender is 2 (female)
        - [age]>=18                    -> Show field if age >= 18
        - [pregnant]='1' and [gender]='2'

        The field is SHOWN when the condition holds, so it should be BLANK when
        the condition does NOT hold. We store ``Not(<parsed condition>)`` with a
        must_be_blank action.
        """
        if not branching_logic or not isinstance(branching_logic, str) or not branching_logic.strip():
            return None

        parsed = parse_redcap_expression(branching_logic.strip())

        return ConditionalRule(
            rule_id=f"{field_name}_show_if",
            rule_type="show_if",
            condition=Not(parsed),  # blank if NOT shown
            action="must_be_blank",
            affected_fields=[field_name],
            description=f"Skip field '{field_name}' when condition is not met: {branching_logic}",
            source=branching_logic,
            severity="warning",
            confidence=1.0,
        )

    def _parse_business_rule(self, field_name: str, rule_text: str) -> Optional[ConditionalRule]:
        """
        Parse a business rule text into a ConditionalRule.

        Recognizes common patterns:
        - "If male, skip this field"
        - "Required for female subjects"
        - "Only if age >= 18"
        """
        if not rule_text or not isinstance(rule_text, str):
            return None

        rule_lower = rule_text.lower()

        # Pattern: "if male" + "skip/blank"
        if "male" in rule_lower and ("skip" in rule_lower or "blank" in rule_lower):
            if "if" in rule_lower and "male" in rule_lower:
                return ConditionalRule(
                    rule_id=f"{field_name}_male_skip",
                    rule_type="skip_if",
                    condition=Compare("gender", "in", ["male", "m", "1"]),
                    action="must_be_blank",
                    affected_fields=[field_name],
                    description=f"Skip {field_name} for male subjects",
                    source=rule_text,
                    severity="error",
                    confidence=0.9,
                )

        # Pattern: "if female" + "required"
        if "female" in rule_lower and "required" in rule_lower:
            return ConditionalRule(
                rule_id=f"{field_name}_female_required",
                rule_type="required_if",
                condition=Compare("gender", "in", ["female", "f", "2"]),
                action="must_be_filled",
                affected_fields=[field_name],
                description=f"Require {field_name} for female subjects",
                source=rule_text,
                severity="error",
                confidence=0.9,
            )

        # Pattern: "only if age >= X"
        age_match = re.search(r'age\s*>=\s*(\d+)', rule_lower)
        if age_match:
            age_threshold = int(age_match.group(1))
            return ConditionalRule(
                rule_id=f"{field_name}_age_threshold",
                rule_type="show_if",
                condition=Not(Compare("age", "ge", age_threshold)),
                action="must_be_blank",
                affected_fields=[field_name],
                description=f"Skip {field_name} if age < {age_threshold}",
                source=rule_text,
                severity="warning",
                confidence=0.8,
            )

        return None

    def _parse_fhir_enable_when(self, field_name: str, enable_when: Dict) -> Optional[ConditionalRule]:
        """
        Parse a FHIR enableWhen structure into a ConditionalRule.

        FHIR enableWhen example:
        {"question": "gender", "operator": "=", "answerString": "female"}
        """
        try:
            question = enable_when.get('question', '')
            operator = enable_when.get('operator', '=')
            answer = (enable_when.get('answerString')
                      or enable_when.get('answerInteger')
                      or enable_when.get('answerBoolean')
                      or enable_when.get('answerCoding', {}).get('code', ''))

            if not question or answer is None:
                return None

            if operator == 'exists':
                # exists=true -> field enabled when question is answered (not blank)
                enable_cond: Condition = (
                    Compare(question, "not_blank") if answer else Compare(question, "is_blank")
                )
            else:
                op = {"=": "eq", "!=": "ne", ">": "gt", "<": "lt", ">=": "ge", "<=": "le"}.get(operator, "eq")
                if op in _NUMERIC_OPS:
                    enable_cond = Compare(question, op, answer)
                else:
                    enable_cond = Compare(question, op, answer)

            return ConditionalRule(
                rule_id=f"{field_name}_enable_when",
                rule_type="show_if",
                condition=Not(enable_cond),  # blank if NOT enabled
                action="must_be_blank",
                affected_fields=[field_name],
                description=f"Enable {field_name} when {question} {operator} {answer}",
                source=json.dumps(enable_when),
                severity="warning",
                confidence=1.0,
            )

        except Exception as e:
            logger.warning(f"Failed to parse FHIR enableWhen: {e}")
            return None

    def _convert_natural_language_condition(self,
                                            condition_text: str,
                                            field_name: str = "") -> Condition:
        """
        Convert a natural-language condition into a structured Condition.

        Handles patterns like:
        - "gender is male"   -> Compare('gender','in',['male','m','1'])
        - "age >= 18"        -> Compare('age','ge',18)
        - "pregnant is yes"  -> Compare('pregnant','in',['yes','y','1','true'])
        - "status != active" -> Compare('status','ne','active')
        - "notes contains x"  -> Compare('notes','contains','x')

        Fail-closed: unparseable text becomes Const(False).
        """
        if not condition_text or not isinstance(condition_text, str):
            return Const(False)

        text = condition_text.lower().strip()

        # Pattern: "field != value" / "field is not value" / "field not equal to value"
        not_equal = re.search(r'(\w+)\s+(?:!=|not equal to|is not)\s+(.+)', text)
        if not_equal:
            return Compare(not_equal.group(1).strip(), "ne", not_equal.group(2).strip())

        # Pattern: "field contains value" / "field includes value"
        contains = re.search(r'(\w+)\s+(?:contains|includes)\s+(.+)', text)
        if contains:
            return Compare(contains.group(1).strip(), "contains", contains.group(2).strip())

        # Pattern: "field >= value" (and >, <=, <)
        comparison = re.search(r'(\w+)\s*(>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)', text)
        if comparison:
            field_nm = comparison.group(1).strip()
            op = {">": "gt", "<": "lt", ">=": "ge", "<=": "le"}[comparison.group(2)]
            num = comparison.group(3)
            value = float(num) if "." in num else int(num)
            return Compare(field_nm, op, value)

        # Pattern: "field is value" / "field = value"
        is_match = re.search(r'(\w+)\s+(?:is|==|=)\s+(.+)', text)
        if is_match:
            fld = is_match.group(1).strip()
            value = is_match.group(2).strip()
            if value in ('male', 'm'):
                return Compare(fld, "in", ["male", "m", "1"])
            if value in ('female', 'f'):
                return Compare(fld, "in", ["female", "f", "2"])
            if value in ('yes', 'y', 'true'):
                return Compare(fld, "in", ["yes", "y", "1", "true"])
            if value in ('no', 'n', 'false'):
                return Compare(fld, "in", ["no", "n", "0", "false"])
            return Compare(fld, "eq", value)

        logger.warning(f"Could not parse natural language condition: {condition_text}")
        return Const(False)

    def _extract_llm_rules(self, field_def: Dict) -> List[ConditionalRule]:
        """
        Extract conditional rules from an LLM-parsed field definition.

        Processes ``conditional_rules`` extracted by the LLM during dictionary
        parsing, converting each condition_text into a structured Condition.
        """
        rules: List[ConditionalRule] = []
        field_name = field_def.get('field_name', '')

        if not field_name:
            return rules

        conditional_rules = field_def.get('conditional_rules', [])
        if not isinstance(conditional_rules, list):
            return rules

        for i, rule_data in enumerate(conditional_rules):
            if not isinstance(rule_data, dict):
                continue

            try:
                rule_type = rule_data.get('rule_type', 'skip_if')
                condition_text = rule_data.get('condition_text', '')
                action = rule_data.get('action', 'must_be_blank')
                affected_fields = rule_data.get('affected_fields', [field_name])

                if not condition_text:
                    continue

                condition = self._convert_natural_language_condition(condition_text, field_name)

                severity = "warning" if rule_type in ["show_if", "allowed_if"] else "error"

                rule = ConditionalRule(
                    rule_id=f"{field_name}_llm_{i}",
                    rule_type=rule_type,
                    condition=condition,
                    action=action,
                    affected_fields=affected_fields if isinstance(affected_fields, list) else [field_name],
                    description=f"LLM-extracted: {condition_text}",
                    source=f"LLM extraction: {condition_text}",
                    severity=severity,
                    confidence=0.85,
                )

                rules.append(rule)
                logger.debug(f"Extracted LLM rule: {rule.description}")

            except Exception as e:
                logger.warning(f"Failed to process LLM rule for {field_name}: {e}")
                continue

        return rules
