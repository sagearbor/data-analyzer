# Real Data Dictionary Test Files

This directory contains real-world data dictionaries for testing.

## Adding New Dictionaries

When you find REDCap or FHIR dictionaries to test with:

1. **Anonymize** - Remove any sensitive information
2. **Name clearly** - Use descriptive filenames
3. **Document** - Add entry below describing the file

## Naming Convention

Use this format: `{format}_{project}_{optional_version}.{ext}`

Examples:
- `redcap_diabetes_study_v2.csv`
- `fhir_covid_screening.json`
- `redcap_oncology_trial.csv`

## Files in This Directory

### REDCap Dictionaries
*(Add your files here with descriptions)*

Example:
```
- redcap_trial_xyz.csv
  - Source: XYZ Clinical Trial
  - Fields: 45
  - Logic rules: 12 branching logic patterns
  - Notes: Complex nested conditions, good for testing
```

### FHIR Questionnaires
*(Add your files here with descriptions)*

Example:
```
- fhir_screening_form.json
  - Source: Patient screening form
  - Items: 30
  - EnableWhen rules: 8
  - Notes: Multi-level conditional logic
```

## What to Test

When adding a new dictionary, test:
1. **Parsing** - Does LLM extract all fields correctly?
2. **Logic extraction** - Are conditional rules identified?
3. **Code generation** - Does generated code work?
4. **Validation** - Does it catch violations correctly?

## .gitignore

These files are in `.gitignore` to prevent accidentally committing real data.
If a file is safe to commit (fully anonymized), remove it from `.gitignore`.
