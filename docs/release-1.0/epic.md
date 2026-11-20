Get the Data Analyzer to its first production release, with the work, the tests and the release documents all on record.

# What the tool does
A person uploads a data file (CSV, JSON, Excel or text) and, if they have one, a data dictionary. The tool checks the data against the dictionary and against general quality rules (types, ranges, allowed values, missing values, duplicates) and shows what it found. An AI model can be used to read a dictionary that is not in a structured format. The same checks can be called from code through an API.

# Scope of release 1.0
- Security review (two independent reviews: Claude Code and Codex) and fixes for what they find
- About page with ready-to-copy blocks so developers can call the tool from code
- A choice of AI model on the page, limited to models the server allows
- The programmatic JSON API (built under ST-358 to ST-365) merged into the release
- A full test of the site, recorded as test cases with results
- The SDLC release documents, in Confluence under SDLC / Data Analyzer
- Deployment to the development environment and a check of the deployed site

# Where things are
- Code: GitHub repository sagearbor/data-analyzer, branch release/1.0-sdlc
- Documents: Confluence space ADP, folder SDLC / Data Analyzer
- Tests: Zephyr, in this Jira project (switched on 20 Nov 2026)

# Notes
- Work is being done with Claude Code on behalf of Sage Arbor. A person reviews and approves; nothing here is signed or approved by the AI.
- The only environment today is the development one (Azure Container Apps, reachable on the Duke VPN only). There is no separate production environment yet; standing one up is outside this epic unless added.
- Each issue is moved to In Progress and Done at the time the work actually starts and ends, so the issue history can be used as a timeline.
