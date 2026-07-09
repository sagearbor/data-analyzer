# Named Program Cache System - Implementation Checklist

## Overview
This checklist guides the implementation of Feature 1 (Named Program Cache System) using the comprehensive test suite that has been created.

## Files Created (Tests)

- [x] `tests/test_program_cache.py` - 50 comprehensive tests (1,153 lines)
- [x] `tests/TEST_PROGRAM_CACHE_README.md` - Test documentation
- [x] `tests/RUN_PROGRAM_CACHE_TESTS.sh` - Automated test runner
- [x] `tests/PROGRAM_CACHE_TEST_SUMMARY.txt` - Quick reference summary

## Files to Implement (Source Code)

### 1. src/program_cache.py

**ValidationProgram Dataclass** (guided by 3 tests):
- [ ] Create dataclass with all required fields
- [ ] Implement `to_dict()` method for serialization
- [ ] Implement `from_dict()` classmethod for deserialization
- [ ] Set proper default values (aliases=[], use_count=0, status="active")

**ProgramDatabase Class** (guided by 25 tests):
- [ ] Implement `__init__()` - database initialization
  - [ ] Create SQLite database with WAL mode
  - [ ] Create tables: programs, aliases, execution_history
  - [ ] Create programs directory
- [ ] Implement `save_program()` - save ValidationProgram to database
- [ ] Implement `load_program()` - load by ID, name, or alias
- [ ] Implement `search_programs()` - search with filters
- [ ] Implement `list_all_programs()` - list ordered by use_count
- [ ] Implement `create_alias()` - create program alias
- [ ] Implement `delete_alias()` - remove alias
- [ ] Implement `increment_use_count()` - track usage
- [ ] Implement `record_execution()` - log execution history
- [ ] Implement `delete_program()` - soft delete with admin password
- [ ] Implement `restore_program()` - restore deleted program
- [ ] Implement `_verify_admin()` - verify ADMIN_PW environment variable

### 2. src/program_manager.py

**ProgramManager Class** (guided by 17 tests):
- [ ] Implement `__init__()` - initialize with database path
- [ ] Implement `_generate_name()` - create YYYYMMDD-HHMMSS-Description format
- [ ] Implement `_clean_name()` - sanitize and limit length
- [ ] Implement `_detect_format()` - detect dictionary format
  - [ ] REDCap CSV detection
  - [ ] FHIR JSON detection
  - [ ] CDISC ODM detection
  - [ ] Unknown fallback
- [ ] Implement `_generate_description()` - detect domain
  - [ ] Clinical domain keywords
  - [ ] Employee domain keywords
  - [ ] Generic fallback
- [ ] Implement `create_program_from_dictionary()` - LLM-based program creation
- [ ] Implement `find_or_create_program()` - deduplication logic
- [ ] Implement `execute_program()` - run validation code

## Testing Workflow (TDD Approach)

### Phase 1: ValidationProgram
```bash
# Run ValidationProgram tests
pytest tests/test_program_cache.py::TestValidationProgram -v

# Expected: 3 tests, 0 passed (import errors)
# After implementation: 3 tests, 3 passed
```

### Phase 2: ProgramDatabase
```bash
# Run ProgramDatabase tests
pytest tests/test_program_cache.py::TestProgramDatabase -v

# Expected: 25 tests, 0 passed (import errors)
# After implementation: 25 tests, 25 passed
```

### Phase 3: ProgramManager
```bash
# Run ProgramManager tests
pytest tests/test_program_cache.py::TestProgramManager -v

# Expected: 17 tests, 0 passed (import errors)
# After implementation: 17 tests, 17 passed
```

### Phase 4: Integration
```bash
# Run Integration tests
pytest tests/test_program_cache.py::TestIntegration -v

# Expected: 5 tests, 0 passed (integration issues)
# After implementation: 5 tests, 5 passed
```

### Phase 5: Full Suite
```bash
# Run all tests with coverage
./tests/RUN_PROGRAM_CACHE_TESTS.sh

# Expected: 50 tests, 50 passed, >90% coverage
```

## Implementation Tips

### Use Test Failures as Guide
1. Run test
2. Read failure message
3. Implement minimal code to pass test
4. Refactor if needed
5. Move to next test

### Example Test-Driven Workflow
```bash
# Start with first test
pytest tests/test_program_cache.py::TestValidationProgram::test_validation_program_defaults -v

# Failure shows what's needed:
# ModuleNotFoundError: No module named 'src.program_cache'

# Create src/program_cache.py with minimal ValidationProgram
# Re-run test until it passes

# Move to next test
pytest tests/test_program_cache.py::TestValidationProgram::test_validation_program_to_dict -v

# Repeat process
```

### Mock LLM Calls
- Tests already mock Azure OpenAI
- No real API calls during testing
- Use `@patch('src.program_manager.AzureOpenAI')`

### Security Implementation
- Read ADMIN_PW from environment variable
- Raise PermissionError for invalid passwords
- Include error message: "Invalid admin password"

### Database Schema
```sql
CREATE TABLE programs (
    program_id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    dictionary_source TEXT,
    dictionary_hash TEXT,
    dictionary_format TEXT,
    generated_code TEXT,
    schema TEXT,  -- JSON
    conditional_rules TEXT,  -- JSON
    created_by TEXT,
    created_at TEXT,
    last_modified_at TEXT,
    num_fields INTEGER,
    num_basic_rules INTEGER,
    num_logic_rules INTEGER,
    use_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'active'
);

CREATE TABLE aliases (
    alias_name TEXT PRIMARY KEY,
    program_id TEXT,
    created_by TEXT,
    created_at TEXT,
    FOREIGN KEY (program_id) REFERENCES programs(program_id)
);

CREATE TABLE execution_history (
    execution_id TEXT PRIMARY KEY,
    program_id TEXT,
    executed_by TEXT,
    executed_at TEXT,
    input_file TEXT,
    num_records INTEGER,
    num_errors INTEGER,
    execution_time_seconds REAL,
    FOREIGN KEY (program_id) REFERENCES programs(program_id)
);
```

## Quality Gates

Before considering feature complete:
- [ ] All 50 tests pass
- [ ] Code coverage > 90%
- [ ] No security vulnerabilities (ADMIN_PW required for destructive operations)
- [ ] All public methods have docstrings
- [ ] Error messages are clear and actionable
- [ ] Database operations are atomic (use transactions)
- [ ] LLM responses are validated before use

## Next Steps After Implementation

1. **Run full test suite**: `./tests/RUN_PROGRAM_CACHE_TESTS.sh`
2. **Review coverage report**: Open `htmlcov/index.html`
3. **Security review**: Verify admin password protection
4. **Integration testing**: Test with real dictionaries (manual)
5. **Performance testing**: Test with large dictionaries
6. **Documentation**: Add usage examples to README.md

## Files Reference

All test files are in: `/home/scb2/PROJECTS/gitRepos-wsl/data-analyzer/tests/`

- Test suite: `test_program_cache.py`
- Documentation: `TEST_PROGRAM_CACHE_README.md`
- Test runner: `RUN_PROGRAM_CACHE_TESTS.sh`
- Summary: `PROGRAM_CACHE_TEST_SUMMARY.txt`

## Support

If tests fail unexpectedly:
1. Check test error message for specific issue
2. Review test docstring for expected behavior
3. Verify environment variables are set
4. Check database permissions
5. Ensure tmp_path fixtures work correctly

For questions about test expectations:
- Each test has a docstring explaining what it tests
- Test names follow pattern: `test_<what>_<scenario>`
- Fixtures provide sample data and configuration
