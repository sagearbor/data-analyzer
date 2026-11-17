"""
Example: Calling the Data Analyzer REST API from another program (ST-361)

A small client for the endpoints a program needs:

    health()        GET  /api/v1/health          (no key)
    version()       GET  /api/v1/version         (no key)
    analyze_file()  POST /api/v1/analyze         (multipart file upload)
    analyze_json()  POST /api/v1/analyze/json    (dataset inline in a JSON body)

Base URL and API key come from the arguments, or from the environment:

    DATA_ANALYZER_API_URL   default http://127.0.0.1:8000
    DATA_ANALYZER_API_KEY   sent as the X-API-Key header when set

Copy this file into your project, or run the demo against a running server:

    python api_server.py                 # in one terminal
    python examples/api_client.py        # in another

The demo data is synthetic. Requires `requests` (already in requirements.txt).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 60


class ApiError(Exception):
    """A non-2xx reply from the API.

    Attributes:
        status_code: HTTP status (400, 401, 403, 413, 422, 429, ...)
        payload: The parsed JSON error body, normally the API's ErrorResponse
            ({"error", "detail", "timestamp", "request_id"}); {} if not JSON
    """

    def __init__(self, status_code: int, payload: Dict[str, Any]):
        self.status_code = status_code
        self.payload = payload
        super().__init__(f"HTTP {status_code}: {payload.get('error') or payload}")


def _base_url(base_url: Optional[str]) -> str:
    return (base_url or os.getenv("DATA_ANALYZER_API_URL") or DEFAULT_BASE_URL).rstrip("/")


def _headers(api_key: Optional[str]) -> Dict[str, str]:
    key = api_key or os.getenv("DATA_ANALYZER_API_KEY")
    return {"X-API-Key": key} if key else {}


def _result(response) -> Dict[str, Any]:
    """Return the JSON body of a 2xx reply, raise ApiError otherwise."""
    if 200 <= response.status_code < 300:
        return response.json()
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    raise ApiError(response.status_code, payload if isinstance(payload, dict) else {})


def health(base_url: Optional[str] = None, session=None) -> Dict[str, Any]:
    """GET /api/v1/health - service status and version. No API key needed."""
    http = session or requests
    return _result(http.get(f"{_base_url(base_url)}/api/v1/health", timeout=DEFAULT_TIMEOUT))


def version(base_url: Optional[str] = None, session=None) -> Dict[str, Any]:
    """GET /api/v1/version - {"name", "version"}. No API key needed."""
    http = session or requests
    return _result(http.get(f"{_base_url(base_url)}/api/v1/version", timeout=DEFAULT_TIMEOUT))


def analyze_file(
    path: str,
    schema: Optional[Dict[str, Any]] = None,
    rules: Optional[Dict[str, Any]] = None,
    min_rows: int = 1,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    session=None,
) -> Dict[str, Any]:
    """POST /api/v1/analyze - upload a file (.csv, .json, .tsv/.txt, .xlsx/.xls).

    Parameters:
        path: File to upload.
        schema: Optional column-type map, e.g. {"age": "int"}.
        rules: Optional rules, e.g. {"age": {"min": 0, "max": 120}}.
        min_rows: Minimum required row count.
        base_url, api_key: Override the environment variables.
        session: Optional requests.Session (or compatible) to send with.

    Returns:
        The quality report: summary, issues, recommendations, quality_checks,
        summary_stats.
    """
    http = session or requests
    # The multipart route takes schema/rules as JSON-encoded form fields.
    form = {"min_rows": str(min_rows)}
    if schema is not None:
        form["schema"] = json.dumps(schema)
    if rules is not None:
        form["rules"] = json.dumps(rules)
    with open(path, "rb") as handle:
        response = http.post(
            f"{_base_url(base_url)}/api/v1/analyze",
            files={"data_file": (Path(path).name, handle)},
            data=form,
            headers=_headers(api_key),
            timeout=DEFAULT_TIMEOUT,
        )
    return _result(response)


def analyze_json(
    data_content: str,
    fmt: str = "csv",
    encoding: str = "text",
    schema: Optional[Dict[str, Any]] = None,
    rules: Optional[Dict[str, Any]] = None,
    min_rows: int = 1,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    session=None,
) -> Dict[str, Any]:
    """POST /api/v1/analyze/json - send the dataset inline in a JSON body.

    Parameters:
        data_content: CSV/TSV/JSON text, or base64 when encoding="base64".
        fmt: "csv", "tsv" or "json".
        encoding: "text" or "base64".
        schema, rules, min_rows, base_url, api_key, session: as analyze_file().

    Returns:
        The same quality report as analyze_file().
    """
    http = session or requests
    body: Dict[str, Any] = {
        "data_content": data_content,
        "format": fmt,
        "encoding": encoding,
        "min_rows": min_rows,
    }
    if schema is not None:
        body["schema"] = schema
    if rules is not None:
        body["rules"] = rules
    response = http.post(
        f"{_base_url(base_url)}/api/v1/analyze/json",
        json=body,
        headers=_headers(api_key),
        timeout=DEFAULT_TIMEOUT,
    )
    return _result(response)


if __name__ == "__main__":
    import tempfile

    # Tiny synthetic dataset: row 3 has a quantity outside the 0-100 rule.
    demo_csv = "widget_id,colour,quantity\n1,red,10\n2,blue,20\n3,green,950\n"
    demo_rules = {"quantity": {"min": 0, "max": 100}}

    try:
        print("health :", health()["status"])
        print("version:", version()["version"])

        report = analyze_json(demo_csv, rules=demo_rules)
        print("analyze_json:", report["summary"]["issues_found"], "issue(s)")
        for issue in report["issues"]:
            print("  -", issue["message"])

        with tempfile.TemporaryDirectory() as tmp:
            demo_path = Path(tmp) / "widgets.csv"
            demo_path.write_text(demo_csv)
            report = analyze_file(str(demo_path), rules=demo_rules)
        print("analyze_file:", report["summary"]["issues_found"], "issue(s)")
    except ApiError as exc:
        raise SystemExit(f"API error: {exc}")
    except requests.ConnectionError:
        raise SystemExit(
            f"Could not reach {_base_url(None)} - start the server with "
            "`python api_server.py` or set DATA_ANALYZER_API_URL."
        )
