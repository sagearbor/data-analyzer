Test the whole site before release, and keep the tests and their results on record.

# Acceptance criteria
1. Automated tests cover every route the service offers and every user path on the page: upload a data file, upload a dictionary, run the analysis, read the results, the About page, the JSON API, errors.
2. A browser test runs the main paths against the deployed site.
3. Each test is described in plain words (objective, precondition, steps, expected result) and recorded in Zephyr with a result, linked to the story it checks.
4. Failures are raised as Jira issues, fixed, and re-run until the run is clean.
5. Results are counted honestly: tests that did not run are shown as not run.

Zephyr was switched on for the ADP project on 20 Nov 2026. Test cases are created in Draft status; a person marks them Approved after reading the descriptions.
