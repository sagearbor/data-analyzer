# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL: Agent Delegation Philosophy

**YOU MUST PROACTIVELY DELEGATE TO AGENTS TO CONSERVE CONTEXT.**

As the main Claude instance, your primary role is to orchestrate and delegate work to specialized subagents. This architecture is designed to:

1. **Preserve your context window** - Offload implementation, testing, security reviews, and code cleanup to agents
2. **Enable parallel execution** - Launch multiple independent agents simultaneously for faster delivery
3. **Provide specialized expertise** - Each agent has deep domain knowledge and follows best practices
4. **Ensure comprehensive coverage** - code-simplifier and security-reviewer systematically scan the entire codebase over multiple iterations

### When to Delegate (Almost Always)

**Delegate to agents for:**
- Any feature implementation (tech-lead-developer)
- All testing work (qc-test-maintainer)
- Security reviews after any code changes (security-reviewer)
- Code cleanup and optimization (code-simplifier)
- UI/UX evaluation (ux-reviewer)

**Handle directly only for:**
- Simple file reads or searches (use Read/Grep tools directly)
- Answering quick questions about the codebase
- Coordinating between agents

### Parallel Agent Execution

**ALWAYS launch agents in parallel when tasks are independent:**
```
# CORRECT - Single message with multiple Task calls
<Task tool call for endpoint A>
<Task tool call for endpoint B>
<Task tool call for endpoint C>

# INCORRECT - Sequential calls that could be parallel
<Task tool call for endpoint A>
[wait for response]
<Task tool call for endpoint B>
[wait for response]
```

Multiple tech-lead-developer instances can work simultaneously on non-conflicting features, endpoints, or modules.

### Iterative Agent Pattern for Full Codebase Coverage

**code-simplifier and security-reviewer are designed for ITERATIVE execution:**

These agents maintain tracking files that enable them to:
- Scan the entire codebase over multiple invocations
- Automatically resume from where they left off
- Track when files were last reviewed
- Detect file changes and trigger re-reviews
- Prioritize sections that need attention

**DO NOT try to scan the entire codebase in a single agent call.** Instead:
1. Call the agent multiple times
2. Each call processes a manageable chunk
3. The agent reads its tracking file to determine what to scan next
4. The agent updates the tracking file after each scan

This pattern prevents context overflow while ensuring comprehensive coverage.

## IMPORTANT: Developer Checklist

**All AI agents MUST check and update `developer_checklist.yaml` when working on this project.**

The checklist contains:
- Development phases with dependencies
- Task statuses (TODO, INPROGRESS, DONE)
- Priority levels and effort estimates
- Known bugs and technical debt

When working on tasks:
1. Check `developer_checklist.yaml` for related tasks
2. Update status to INPROGRESS when starting work
3. Add detailed notes about progress
4. Mark as DONE only when fully complete
5. Update the `last_updated` field

Example of updating a task:
```yaml
status: INPROGRESS
notes: |
  INPROGRESS:
  - Completed: Basic implementation
  - Remaining: Tests and documentation
  - Blocked: Waiting for dependency X
```

## Project Overview

This is a data quality analysis tool with Azure deployment capabilities that consists of:
- **MCP Server** (`mcp_server.py`): Model Context Protocol server for data analysis
- **Web Application** (`web_app.py`): Streamlit-based web interface
- **Azure Deployment**: Container-based deployment to Azure Container Apps

The project analyzes structured data (currently CSV, extensible for JSON/Excel/Parquet) performing quality checks like data type validation, value range checks, missing value analysis, and duplicate detection.

## Development Commands

### Local Development
```bash
# Set up virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt       # Web app dependencies (includes LLM support)
pip install -r mcp_requirements.txt   # MCP server dependencies

# Run MCP server locally
python mcp_server.py

# Run web application with LLM support
# Option 1: Use the launch script (recommended)
./run_app.sh

# Option 2: Use Python wrapper
python run_streamlit.py

# Option 3: Manual activation and run
source venv/bin/activate && streamlit run web_app.py
```

**IMPORTANT:** Always use one of the above methods to ensure LLM functionality works correctly. Do NOT run `streamlit run web_app.py` directly without activating the virtual environment first.

**NOTE:** The application runs on port 3002 by default (configured in `run_app.sh`) for NGINX reverse proxy compatibility. When running manually, use: `streamlit run web_app.py --server.port 3002`

### Environment Configuration

The application uses the `APP_ENV` environment variable to control the environment warning banner:

- **`APP_ENV=dev`** (default): Red banner - "Development Environment"
- **`APP_ENV=staging`**: Yellow banner - "Staging Environment"
- **`APP_ENV=prod`**: No banner (production)
- **Not set**: Red banner (fail-safe default)

**Local Development:**
1. Copy `.env.example` to `.env`
2. Set `APP_ENV=dev` in your `.env` file
3. The banner will automatically appear at the bottom of the web interface

**Production Deployment:**
- Contact your IT/DevOps team for deployment procedures
- If using containerized deployment, set `APP_ENV` via environment variables:
  - Production: `APP_ENV=prod` (hides banner)
  - Staging: `APP_ENV=staging` (yellow banner)
- See `./scripts/experimental/azure/` for archived Azure deployment scripts (not used in production)

### Docker Development
```bash
# Build and run container (uses port 8002 internally)
docker build -t data-analyzer .
docker run -p 3002:8002 data-analyzer

# With environment variable
docker run -p 3002:8002 -e APP_ENV=dev data-analyzer
```

### Deployment
**Note:** Experimental deployment scripts are in `./scripts/experimental/azure/` but are NOT used in production. Contact your IT/DevOps team for actual deployment procedures.

### Testing
Currently no formal test framework is configured. The README mentions test files that don't exist yet:
- `python -m pytest tests/test_mcp_server.py` (planned)
- `python -m pytest tests/test_web_app.py` (planned)
- `python tests/test_integration.py` (planned)

## Agent Architecture

This repository includes five specialized agents in `.claude/agents/`:

### tech-lead-developer (Blue)
Primary development agent for implementing features and modules. Designed for parallel execution of independent tasks.

**Key Use Cases:**
- Creating independent API endpoints or services (MCP tools, FastAPI endpoints)
- Implementing utility functions and modules (data loaders, quality checkers)
- Building isolated frontend components (Streamlit widgets, dashboard sections)
- Developing features without shared dependencies

**Parallel Development Pattern:**
When you have multiple independent tasks (e.g., three separate data quality checks), launch multiple instances of this agent simultaneously using parallel Task tool calls in a single message.

### qc-test-maintainer (Yellow)
Quality assurance specialist for comprehensive testing.

**Key Responsibilities:**
- Create unit, integration, and end-to-end tests in `tests/` directory
- Maintain and update existing test suites
- Run test validation after changes
- Build mock data and fixtures in `tests/fixtures/`
- Use pytest as the testing framework
- Ensure tests are deterministic, independent, and fast

**Testing Standards:**
- ALL test files go in `tests/` directory (NEVER in project root)
- Use `conftest.py` for shared fixtures
- Store test outputs in `tests/outputs/` (gitignored)
- Run full suite with `pytest tests/`

### security-reviewer (Red)
Security architect focused on threat modeling and vulnerability assessment.

**Key Focus Areas:**
- OWASP Top 10 vulnerabilities
- File upload validation and data handling security (CSV, JSON, Excel, Parquet)
- MCP server endpoint security
- Environment variable and secrets management
- Docker and deployment security configurations
- Data privacy and PII protection in analysis outputs

**Iterative Scanning Pattern:**
The security-reviewer maintains a `.security-review-tracking.json` file to systematically scan the entire codebase over multiple iterations. This enables:
- Comprehensive security coverage without overwhelming context windows
- Tracking which files/modules have been reviewed and when
- Automatic re-scanning when files are modified
- Prioritization of high-risk areas (file uploads, MCP endpoints, data processing)

**When to Use:**
1. **Proactively after implementing features** involving file uploads, data handling, MCP endpoints, or deployment configurations
2. **Periodically for full codebase scans** - The agent will check its tracking file and scan sections that haven't been reviewed recently or have changed since last review
3. **Before deployments** - Full security audit by calling the agent multiple times to cover all modules

**Usage Pattern:**
Call the security-reviewer multiple times to scan the entire codebase in chunks. The agent will automatically track progress and resume from where it left off.

### code-simplifier (Cyan)
Software architect for systematic codebase cleanup and optimization.

**Core Responsibilities:**
- Remove code duplication and orphaned code
- Simplify unnecessary complexity
- Maintain `.code-simplifier-tracking.json` to rotate through codebase
- Run tests before and after changes to ensure nothing breaks
- Work on feature branches, never directly on main

**Iterative Scanning Pattern:**
The code-simplifier maintains a `.code-simplifier-tracking.json` file that tracks:
- When each file/module was last reviewed
- What issues were found and fixed
- What changes are pending developer approval
- A rotation schedule to ensure comprehensive coverage over time

This enables systematic cleanup of the entire codebase without context overload.

**Operational Pattern:**
1. Check `.code-simplifier-tracking.json` for sections due for review or recently modified code
2. Analyze code for duplication, orphaned code, and complexity
3. Make safe changes immediately (unused imports, dead code, simple refactoring)
4. Consult developer for complex refactoring requiring architectural changes
5. Always run relevant tests before committing
6. Update tracking file with review timestamps and findings

**When to Use:**
1. **After completing features** - Clean up newly written code before moving on
2. **Periodically for rotation** - The agent will query its tracking file to find sections that haven't been reviewed recently
3. **Before releases** - Call multiple times to systematically clean the entire codebase
4. **When files change** - Agent checks tracking file against actual file modification times to identify code needing re-review

**Usage Pattern:**
Call the code-simplifier multiple times to systematically rotate through the entire codebase. The agent will automatically track which sections need attention and resume from the appropriate point.

### ux-reviewer (Pink)
User experience specialist for visual appeal and cross-platform usability.

**Review Focus:**
- Visual design assessment (color, typography, spacing, hierarchy)
- Usability evaluation (task flow, affordances, feedback, error handling)
- Cross-platform compatibility (mobile, tablet, desktop)
- Code efficiency (avoid UI bloat, leverage Streamlit features)
- Accessibility (WCAG 2.1 AA compliance)

**When to Use:**
After implementing or updating Streamlit UI components, making layout changes, or adding interactive elements to the dashboard.

## Recommended Agent Workflows

### Feature Development Workflow
1. **tech-lead-developer**: Implement the feature
2. **qc-test-maintainer**: Create comprehensive tests
3. **security-reviewer**: Audit for vulnerabilities
4. **code-simplifier**: Optimize and refactor
5. **ux-reviewer**: Review if UI components involved

### Parallel Development Pattern
For independent tasks that don't share dependencies:
```
"I need three new quality checks: null detection, outlier detection, and pattern matching.
Use tech-lead-developer agents in parallel for each."
```

Launch multiple agents simultaneously by making multiple Task tool calls in a single message.

### Maintenance Workflow (Iterative Pattern)

**IMPORTANT:** code-simplifier and security-reviewer are designed for iterative execution. Call them multiple times to systematically cover the entire codebase.

- **Weekly code cleanup**: Call code-simplifier multiple times - it will query `.code-simplifier-tracking.json` to review modules not checked recently or that have changed
- **Weekly security scans**: Call security-reviewer multiple times - it will query `.security-review-tracking.json` to scan sections due for review
- **After major changes**: Use qc-test-maintainer to run full test suite and update tests
- **Pre-deployment**:
  - Call security-reviewer multiple times to complete full security audit
  - Call code-simplifier multiple times to clean entire codebase
  - Call qc-test-maintainer to verify all tests pass

**Example - Full Codebase Security Audit:**
```
Call security-reviewer (iteration 1 - scans MCP server endpoints)
Call security-reviewer (iteration 2 - scans file upload handlers)
Call security-reviewer (iteration 3 - scans data processing pipeline)
Call security-reviewer (iteration 4 - scans deployment configs)
```

Each call automatically picks up where the previous scan left off by reading the tracking file.

## Architecture

### Core Components

1. **MCP Server** (`mcp_server.py:28-506`): 
   - `DataLoader` class for loading data formats (currently CSV, extensible)
   - `QualityChecker` class for performing data quality validations
   - `QualityPipeline` class for orchestrating checks
   - MCP server implementation with `analyze_data` and `get_data_info` tools

2. **Web Application** (`web_app.py:1-681`):
   - Streamlit interface with file upload, schema editor, rules editor
   - `MCPClient` class for communicating with MCP server (currently simulated)
   - Interactive dashboard for displaying analysis results

3. **Data Quality Checks**:
   - Row count validation (`mcp_server.py:67-83`)
   - Data type validation (`mcp_server.py:85-150`)
   - Value range validation (`mcp_server.py:152-234`)
   - Summary statistics generation (`mcp_server.py:236-262`)

### Key Patterns

- **Extensible Data Loading**: `DataLoader.load_data()` method designed to support multiple formats
- **MCP Protocol**: Server follows Model Context Protocol standards for tool definitions
- **Async Processing**: MCP server uses async/await patterns
- **Streamlit State Management**: Web app uses `st.session_state` for schema/rules configuration

## Configuration

### Environment Variables
- `APP_ENV`: Environment indicator (dev/staging/prod) - controls warning banner
  - `dev`: Red banner with "Development Environment"
  - `staging`: Yellow banner with "Staging Environment"
  - `prod`: No banner
  - Not set: Red banner (fail-safe default)

**Note:** Azure deployment variables archived in `./scripts/experimental/azure/`

### File Formats Supported
- **CSV**: Full support with encoding detection
- **JSON, Excel, Parquet**: Planned (architecture supports extension)

## Important Implementation Notes

- Web app currently simulates MCP calls via `_simulate_mcp_call()` method rather than using actual MCP client
- MCP server supports base64 encoded data and data URLs
- Schema validation supports: int, float, str, bool, datetime types
- Rules validation supports: min/max ranges for numeric data, allowed values for categorical data
- Environment banner powered by `env-banner-python` package (bottom-positioned)

## File Structure
```
.
├── .claude/
│   └── agents/              # Agent definition files (DO NOT modify without understanding impact)
│       ├── tech-lead-developer.md
│       ├── qc-test-maintainer.md
│       ├── security-reviewer.md
│       ├── code-simplifier.md
│       └── ux-reviewer.md
├── .gitignore               # Python-standard gitignore
├── CLAUDE.md                # This file - agent orchestration guidance
├── README.md                # User-facing documentation
├── mcp_server.py            # MCP server implementation
├── web_app.py               # Streamlit web application
├── requirements.txt         # Web app dependencies
├── mcp_requirements.txt     # MCP server dependencies
├── Dockerfile               # Container configuration
├── run_app.sh               # App launcher script (port 3002)
├── .env.example             # Environment variable template
├── developer_checklist.yaml # Task tracking for development
├── scripts/experimental/    # Experimental features (not production)
├── docs/API.md              # Detailed API documentation
└── [Agent tracking files - created at runtime, add to .gitignore]
    ├── .code-simplifier-tracking.json
    └── .security-review-tracking.json
```

## Agent Tracking Files

The code-simplifier and security-reviewer agents maintain tracking files to systematically scan the entire codebase over multiple iterations:

**`.code-simplifier-tracking.json`**
- Tracks when each file/module was last reviewed
- Records issues found, fixed, and pending
- Maintains rotation schedule for comprehensive coverage
- Updates automatically on each agent invocation

**`.security-review-tracking.json`**
- Tracks security review status of all files/modules
- Records vulnerability findings and remediation status
- Prioritizes high-risk areas (file uploads, MCP endpoints, data processing)
- Tracks when files change to trigger re-scanning

**IMPORTANT:** Add these tracking files to your `.gitignore` as they are runtime artifacts specific to each developer's workflow and should not be committed to version control.

## Agent Invocation Best Practices

**Context Conservation (Primary Goal):**
- **ALWAYS delegate to agents** for any non-trivial work to preserve main context window
- Offload implementation, testing, security, and cleanup work to subagents
- Your role as main Claude is to orchestrate, not to implement

**Parallel Execution:**
- **Launch multiple agents in single message** when tasks are independent
- Multiple tech-lead-developer instances can work simultaneously on different features
- Never wait for one agent to finish if another can start immediately

**Iterative Patterns:**
- **Call code-simplifier multiple times** to systematically rotate through entire codebase
- **Call security-reviewer multiple times** for comprehensive security coverage
- Agents automatically track progress via `.code-simplifier-tracking.json` and `.security-review-tracking.json`
- Each invocation picks up where the last one left off

**Agent Communication:**
- Provide clear, complete instructions for what agent should accomplish
- Specify what information agent should return in its final report
- Each agent invocation is stateless - provide full context needed
- For parallel work, ensure tasks are truly independent with no shared dependencies