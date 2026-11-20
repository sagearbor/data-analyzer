Some of the existing automated tests prove little, and a few cannot fail. Found while writing a plain-language description of every test for Zephyr.

# Examples
1. Four tests of the MCP tool path catch every error, so they always pass.
2. Several API tests accept "service unavailable" or a server error as a pass.
3. Some tests are named for one thing and check another (for example ordering, concurrency, format filtering).
4. Six tests need a live AI model or live API and are skipped in the normal run.

# What has been done
Each such test is described honestly in Zephyr (marked as a weak check) rather than hidden. Tests touched by the security fixes were tightened.

# Acceptance criteria
1. Tests that cannot fail are rewritten to check a real result, or removed.
2. Loose pass conditions are replaced with the one expected answer.
3. Mis-named tests are renamed or made to test what the name says.
4. A decision is recorded on when and how the six live tests are run.

Raised during release 1.0 work (epic ADP-48). Posted by Claude Code for Sage Arbor.
