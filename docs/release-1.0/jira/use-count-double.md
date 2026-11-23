Running a saved dictionary program once adds 2 to its use count instead of 1.

# Cause
`ProgramManager.execute_program()` in `src/program_manager.py` calls `increment_use_count()` and then `record_execution()`, and the second call also increments the count.

# Effect
The "times used" figure for saved dictionaries is double the true number. No effect on data checking results.

# Found by
The strengthened test `tests/test_program_cache.py::TestProgramManager::test_execute_program_counts_one_use_per_run` (ADP-89). It is marked as an expected failure until this is fixed.

# Acceptance criteria
1. One run adds exactly 1 to the use count.
2. The expected-failure mark is removed and the test passes.

Raised during release 1.0 work (epic ADP-48). Posted by Claude Code for Sage Arbor.
