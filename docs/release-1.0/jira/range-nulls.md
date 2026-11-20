Range checks did not report violations in any column that also had missing values.

# What was wrong
For a column with a minimum or maximum rule and at least one empty cell, the checker hit an internal error that was silently swallowed. The result was that out-of-range values in that column were never reported, and the report looked clean. This has been in the code since before this release.

# How it was found
A browser test of the main path failed after error handling was tightened; removing the swallowed error exposed the real fault on three of the built-in demo datasets.

# Fix
The checker now reports these violations (commit 1613567). Issue counts on data with missing values will be higher than before - correctly.

# Acceptance criteria
1. A test with a column holding both empty cells and out-of-range values reports the violations.
2. Every built-in demo dictionary analyses with its demo data without an error.
3. Earlier reports produced by the tool for data with missing values are treated as possibly incomplete; the owner decides whether anyone needs to be told.

Raised during release 1.0 work (epic ADP-48). Posted by Claude Code for Sage Arbor.
