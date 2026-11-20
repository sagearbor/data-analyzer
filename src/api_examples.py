"""
Copy-and-paste API examples shown on the About page (ADP-51).

One place for the example text: web_app.py renders these blocks and
tests/test_about_api_examples.py executes the very same strings against a
real local server, so what the page shows is what is tested.

Rules every block follows:
- runs as written once the developer has exported DATA_ANALYZER_API_KEY;
- self-contained: a block that needs a sample file creates it itself, and the
  JSON block needs no file at all;
- curl blocks are ONE line (no backslash continuations);
- Python blocks need only the `requests` package;
- the key is only ever referenced as $DATA_ANALYZER_API_KEY (bash) or
  os.environ["DATA_ANALYZER_API_KEY"] (Python), never as a value. This module
  is pure: it does not import os or read any environment variable.
"""

import re
from typing import Dict, List, Optional, Tuple

PLACEHOLDER_URL = "https://<data-analyzer-api-fqdn>"
LOCAL_URL = "http://localhost:8000"

# A base URL is pasted into shell commands, so only plain URL characters are
# accepted; anything else is refused rather than escaped.
_SAFE_URL = re.compile(r"^https?://[A-Za-z0-9._~:/\-\[\]%]+$")

# (group id, heading, one-line explanation) in display order.
EXAMPLE_GROUPS: List[Tuple[str, str, str]] = [
    ("health", "1. Check the service is up", "No API key needed."),
    ("version", "2. Get the version", "No API key needed."),
    ("analyze_file", "3. Analyze a file",
     "Uploads a CSV as multipart form data. The block writes its own sample file, demo.csv."),
    ("analyze_json", "4. Analyze data sent as JSON",
     "No file needed: the dataset travels inline in the JSON body, with an optional "
     "column-type schema and range rules."),
]

_HEALTH_CURL = 'curl -sS "__URL__/api/v1/health"'

_VERSION_CURL = 'curl -sS "__URL__/api/v1/version"'

_ANALYZE_FILE_CURL = (
    "printf 'age,name\\n30,Jane\\n' > demo.csv && "
    'curl -sS -X POST "__URL__/api/v1/analyze" '
    '-H "X-API-Key: $DATA_ANALYZER_API_KEY" '
    '-F "data_file=@demo.csv;type=text/csv"'
)

_ANALYZE_JSON_CURL = (
    'curl -sS -X POST "__URL__/api/v1/analyze/json" '
    '-H "X-API-Key: $DATA_ANALYZER_API_KEY" '
    '-H "Content-Type: application/json" '
    "-d '{\"data_content\":\"age,name\\n30,Jane\\n150,Bob\\n\",\"format\":\"csv\","
    "\"schema\":{\"age\":\"int\"},\"rules\":{\"age\":{\"min\":0,\"max\":120}}}'"
)

_HEALTH_PYTHON = '''import json
import requests

resp = requests.get("__URL__/api/v1/health", timeout=10)
resp.raise_for_status()
print(json.dumps(resp.json(), indent=2))
'''

_VERSION_PYTHON = '''import json
import requests

resp = requests.get("__URL__/api/v1/version", timeout=10)
resp.raise_for_status()
print(json.dumps(resp.json(), indent=2))
'''

_ANALYZE_FILE_PYTHON = '''import json
import os
import requests

# Create a small sample file (use your own CSV instead).
with open("demo.csv", "w") as f:
    f.write("age,name\\n30,Jane\\n")

headers = {"X-API-Key": os.environ["DATA_ANALYZER_API_KEY"]}
with open("demo.csv", "rb") as f:
    resp = requests.post(
        "__URL__/api/v1/analyze",
        headers=headers,
        files={"data_file": ("demo.csv", f, "text/csv")},
        timeout=60,
    )
resp.raise_for_status()
print(json.dumps(resp.json(), indent=2))
'''

_ANALYZE_JSON_PYTHON = '''import json
import os
import requests

headers = {"X-API-Key": os.environ["DATA_ANALYZER_API_KEY"]}
payload = {
    "data_content": "age,name\\n30,Jane\\n150,Bob\\n",
    "format": "csv",
    "schema": {"age": "int"},                      # optional column types
    "rules": {"age": {"min": 0, "max": 120}},      # optional range rules
}
resp = requests.post("__URL__/api/v1/analyze/json", headers=headers, json=payload, timeout=60)
resp.raise_for_status()
print(json.dumps(resp.json(), indent=2))
'''

_TEMPLATES = [
    ("health", "curl", "Check the service is up (curl)", "bash", _HEALTH_CURL),
    ("health", "python", "Check the service is up (Python)", "python", _HEALTH_PYTHON),
    ("version", "curl", "Get the version (curl)", "bash", _VERSION_CURL),
    ("version", "python", "Get the version (Python)", "python", _VERSION_PYTHON),
    ("analyze_file", "curl", "Analyze a file (curl)", "bash", _ANALYZE_FILE_CURL),
    ("analyze_file", "python", "Analyze a file (Python)", "python", _ANALYZE_FILE_PYTHON),
    ("analyze_json", "curl", "Analyze data sent as JSON (curl)", "bash", _ANALYZE_JSON_CURL),
    ("analyze_json", "python", "Analyze data sent as JSON (Python)", "python", _ANALYZE_JSON_PYTHON),
]


def display_base_url(runtime_url: Optional[str]) -> Tuple[str, str]:
    """Choose the base URL to show in the examples.

    Args:
        runtime_url: Value of DATA_ANALYZER_API_URL as the UI sees it (may be
            empty or None).

    Returns:
        (url, note). ``note`` is a short explanation to show next to the
        examples, or "" when none is needed. The docker-compose service host
        ("://api:") only resolves inside the container network, so it is
        swapped for localhost; a missing or unusable value gives
        PLACEHOLDER_URL.
    """
    url = (runtime_url or "").strip().rstrip("/")
    if not url or not _SAFE_URL.match(url):
        return PLACEHOLDER_URL, (
            "DATA_ANALYZER_API_URL is not set for this UI, so the examples use a "
            "placeholder address - replace it with the API address from the app owner."
        )
    if "://api:" in url:
        return LOCAL_URL, (
            "This UI is running via docker-compose.fullstack.yml; the examples use "
            "localhost:8000, which is the API as seen from your own machine."
        )
    return url, ""


def build_api_examples(base_url: str) -> List[Dict[str, str]]:
    """Build the copy-paste blocks for a given API base URL.

    Args:
        base_url: API root such as "https://host" or "http://localhost:8000";
            a trailing slash is ignored. PLACEHOLDER_URL is accepted too.

    Returns:
        Ordered list of dicts with keys ``id`` (e.g. "analyze_json_curl"),
        ``group`` (e.g. "analyze_json"), ``title``, ``language`` ("bash" or
        "python") and ``code`` (the exact text to display and run).

    Raises:
        ValueError: if base_url contains characters that are unsafe to paste
            into a shell command.
    """
    base = (base_url or "").strip().rstrip("/")
    if base != PLACEHOLDER_URL and not _SAFE_URL.match(base):
        raise ValueError("base_url must be a plain http(s) URL")
    return [
        {
            "id": f"{group}_{kind}",
            "group": group,
            "title": title,
            "language": language,
            "code": template.replace("__URL__", base).strip("\n"),
        }
        for group, kind, title, language, template in _TEMPLATES
    ]
