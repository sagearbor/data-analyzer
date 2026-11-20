Release request for Data Analyzer 1.0, the first production release. This Change Request stands in for the "DS Release Request" described in DS-S-001 step V.A.2, because project ADP does not have that work type.

# What is being released
Data Analyzer checks a data file against a data dictionary and general quality rules (data types, value ranges, allowed values, missing values, duplicates) and reports what it finds. A person uses it through a web page; a program can use it through an API. An AI model may be used to read a dictionary that is not in a structured format: the dictionary text is sent to the model, the data rows are not.

Release 1.0 covers the web page, the API (file upload and JSON), a choice of AI model from a server-controlled list, code examples on the About page, and the fixes from the pre-release security review. Work is tracked under epic ADP-48.

# Preliminary risk assessment
Scored against the Integrations first-release impact and complexity definitions (Confluence space DSQSD) and the classification table in "V.A.3 Making a Preliminary Risk Assessment".

- Impact: Medium. The tool supports data cleaning and review; an error has an indirect effect because data are reviewed by other methods and parties. It would be High if the tool were the only check on critical study data, or were used with personally identifying information.
- Complexity: Medium. Several file formats, moderate data work, two new container applications. Parts of High are present: new infrastructure, a call to an AI model service, an API.
- Risk level: Medium.

This is a preliminary assessment prepared by Claude Code (an AI assistant) for Sage Arbor. The Release Advisory Board makes the final decision. Two assumptions for the board: that "Integrations" is the right solution type for a general data-checking tool, and that DS-S-001 is the right procedure for it.

# Draft Validation Plan
https://dcri.atlassian.net/wiki/spaces/ADP/pages/6259474534/DRAFT+INT.Data+Analyzer+1.0_VP

# Out of order
Development and testing began before this request was written. The Review Date is left empty: a person convenes the board.

# Conditions the board should know about
- ADP-69: files in the public code repository need a privacy and security review.
- ADP-71: the page has no per-user sign-in. ADP-75: the applications share a cloud identity with wide rights. Both need IT.
- Only a development environment exists; there is no separate production environment.
