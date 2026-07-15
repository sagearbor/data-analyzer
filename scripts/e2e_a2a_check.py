#!/usr/bin/env python3
"""
scripts/e2e_a2a_check.py — Live end-to-end proof that the deployed
data-analyzer Cloud Run service is reachable and returns correct results.

STDLIB ONLY — no pandas, no requests, no third-party packages.

Usage
-----
    python3 scripts/e2e_a2a_check.py --url https://<service-url>
    E2E_URL=https://<service-url> python3 scripts/e2e_a2a_check.py

Exit codes
----------
    0  All checks passed.
    1  One or more checks failed (detailed FAIL messages printed).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Tuple

# ---------------------------------------------------------------------------
# Tiny HTTP helpers (urllib only)
# ---------------------------------------------------------------------------


def _get(url: str, *, timeout: float = 30.0) -> Tuple[int, dict | str]:
    """GET *url* and return (status_code, parsed_body).

    Body is parsed as JSON if Content-Type is application/json, otherwise
    returned as a raw string.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            ct = resp.headers.get_content_type() or ""
            if "json" in ct:
                return resp.status, json.loads(raw)
            return resp.status, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace") if exc.fp else ""
        try:
            return exc.code, json.loads(raw)
        except Exception:
            return exc.code, raw
    except urllib.error.URLError as exc:
        raise SystemExit(f"FATAL: Cannot reach {url!r}: {exc.reason}") from exc


def _post_json(url: str, body: dict, *, timeout: float = 60.0) -> Tuple[int, dict | str]:
    """POST *body* as JSON to *url* and return (status_code, parsed_body)."""
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except Exception:
                return resp.status, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace") if exc.fp else ""
        try:
            return exc.code, json.loads(raw)
        except Exception:
            return exc.code, raw
    except urllib.error.URLError as exc:
        raise SystemExit(f"FATAL: Cannot reach {url!r}: {exc.reason}") from exc


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------

_failures: list[str] = []
_passes: list[str] = []


def _ok(label: str, detail: str = "") -> None:
    msg = f"  PASS  {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    _passes.append(label)


def _fail(label: str, detail: str = "") -> None:
    msg = f"  FAIL  {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    _failures.append(f"{label}: {detail}")


def _assert(condition: bool, label: str, detail: str = "") -> bool:
    if condition:
        _ok(label, detail)
    else:
        _fail(label, detail)
    return condition


# ---------------------------------------------------------------------------
# Check 1 — GET /health
# ---------------------------------------------------------------------------

def check_health(base_url: str) -> None:
    print("\n[1/3] GET /health")
    url = f"{base_url}/health"
    status, body = _get(url)
    if not _assert(status == 200, "/health returns HTTP 200", f"got {status}"):
        return
    if not isinstance(body, dict):
        _fail("/health body is JSON dict", f"got {type(body).__name__}: {body!r:.80}")
        return
    _assert(body.get("status") == "ok", '/health body.status == "ok"', f"got {body.get('status')!r}")
    _assert("a2a_backend" in body, "/health body contains 'a2a_backend'", str(body))


# ---------------------------------------------------------------------------
# Check 2 — GET /.well-known/agent.json
# ---------------------------------------------------------------------------

def check_agent_card(base_url: str) -> None:
    print("\n[2/3] GET /.well-known/agent.json")
    url = f"{base_url}/.well-known/agent.json"
    status, body = _get(url)
    if not _assert(status == 200, "/.well-known/agent.json returns HTTP 200", f"got {status}"):
        return
    if not isinstance(body, dict):
        _fail("agent.json body is JSON dict", f"got {type(body).__name__}")
        return
    expected_url = "/a2a/data-analyzer"
    card_url: Any = body.get("url")
    _assert(
        card_url == expected_url,
        f"agent.json url == {expected_url!r}",
        f"got {card_url!r}",
    )
    _assert("name" in body, "agent.json has 'name' field", str(body.get("name")))
    _assert("version" in body, "agent.json has 'version' field", str(body.get("version")))


# ---------------------------------------------------------------------------
# Check 3 — A2A call via a2a_runtime.http_client.call_agent_http
# ---------------------------------------------------------------------------

def check_a2a_call(base_url: str) -> None:
    """Build a tiny CSV, base64-encode it, and call the A2A endpoint.

    The import of call_agent_http is done here so the rest of the script
    still runs (and checks 1 & 2 still report) if the repo is not on sys.path.
    """
    print("\n[3/3] A2A call via a2a_runtime.http_client.call_agent_http")

    # Ensure the repo root is importable.
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    try:
        from a2a_runtime.http_client import call_agent_http  # noqa: PLC0415
    except ImportError as exc:
        _fail("import a2a_runtime.http_client", str(exc))
        return

    # Tiny CSV with an intentional age outlier (200) to trigger a quality flag.
    csv_text = "id,age\n1,30\n2,40\n3,200\n"
    b64_csv = base64.b64encode(csv_text.encode("utf-8")).decode("ascii")

    payload = {
        "operation": "analyze",
        "data_content": b64_csv,
        "file_format": "csv",
    }

    try:
        result = call_agent_http(
            base_url,
            "data-analyzer",
            payload,
            correlation_id="e2e-1",
        )
    except Exception as exc:
        _fail("call_agent_http completes without exception", str(exc))
        return

    if not _assert(isinstance(result, dict), "A2A response is a dict", type(result).__name__):
        return

    # The quality pipeline always returns at least one of these top-level keys.
    EXPECTED_KEYS = {"checks", "summary_stats", "row_count", "error"}
    found_keys = EXPECTED_KEYS & set(result.keys())
    _assert(
        bool(found_keys),
        f"result contains quality-pipeline keys (one of {sorted(EXPECTED_KEYS)})",
        f"actual keys: {sorted(result.keys())}",
    )

    # Ensure it is not a pure error response.
    _assert(
        "error" not in result or "checks" in result or "row_count" in result,
        "result is not an error-only response",
        str(result)[:120],
    )

    print(f"       result keys: {sorted(result.keys())}")


# ---------------------------------------------------------------------------
# curl example
# ---------------------------------------------------------------------------

def print_curl_example(base_url: str) -> None:
    csv_text = "id,age\n1,30\n2,40\n3,200\n"
    b64_csv = base64.b64encode(csv_text.encode("utf-8")).decode("ascii")
    payload = json.dumps({"data_content": b64_csv, "file_format": "csv"}, indent=2)
    print("\n---- curl example (POST /analyze) ----")
    print(f"curl -s -X POST {base_url}/analyze \\")
    print("     -H 'Content-Type: application/json' \\")
    print(f"     -d '{payload}'")
    print("--------------------------------------")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> str:
    """Return the service base URL from --url or $E2E_URL."""
    parser = argparse.ArgumentParser(
        description="Live end-to-end A2A check for the deployed data-analyzer service."
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("E2E_URL", ""),
        help="Base URL of the deployed service (e.g. https://data-analyzer-xxxx.run.app). "
             "Also read from $E2E_URL.",
    )
    args = parser.parse_args()
    if not args.url:
        parser.error("Provide --url or set the E2E_URL environment variable.")
    return args.url.rstrip("/")


def main() -> None:
    base_url = _parse_args()
    print(f"Target: {base_url}")

    check_health(base_url)
    check_agent_card(base_url)
    check_a2a_call(base_url)
    print_curl_example(base_url)

    # Summary
    total = len(_passes) + len(_failures)
    print(f"\n{'='*50}")
    if not _failures:
        print(f"PASS — all {total} checks passed.")
        print("="*50)
        sys.exit(0)
    else:
        print(f"FAIL — {len(_failures)}/{total} checks failed:")
        for f in _failures:
            print(f"  - {f}")
        print("="*50)
        sys.exit(1)


if __name__ == "__main__":
    main()
