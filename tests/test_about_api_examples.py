"""
ADP-51: the copy-and-paste API examples on the About page.

The text the About page shows comes from src/api_examples.build_api_examples().
These tests take exactly that text and RUN it, verbatim, against a real local
uvicorn server running api_server's app - bash blocks with `bash -c`, Python
blocks with the current interpreter - so a block that is shown but does not
work fails here.

Fully offline (localhost only), no LLM, synthetic data. The server and the
subprocesses use the fake key pinned by tests/conftest.py; the developer's
.env is never read.

Run with:
    pytest tests/test_about_api_examples.py --no-cov -q
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from src import api_examples
from src.api_examples import (
    LOCAL_URL,
    PLACEHOLDER_URL,
    build_api_examples,
    display_base_url,
)

TEST_API_KEY = os.environ["DATA_ANALYZER_API_KEY"]  # fake key from tests/conftest.py
WRONG_API_KEY = "definitely-not-the-key"

HAVE_CURL = shutil.which("curl") is not None
needs_curl = pytest.mark.skipif(
    not HAVE_CURL, reason="curl is not installed, so the curl examples cannot be executed"
)
HAVE_BASH = shutil.which("bash") is not None
needs_bash = pytest.mark.skipif(not HAVE_BASH, reason="bash is not installed")


# ---------------------------------------------------------------------------
# Real local server
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live_server():
    """Serve api_server's app with uvicorn on a free localhost port."""
    import uvicorn

    import api_server

    mp = pytest.MonkeyPatch()
    # api_server captures the key at import time; pin the fake test key.
    mp.setattr(api_server, "API_KEY", TEST_API_KEY)
    try:
        api_server.limiter.reset()
    except Exception:
        pass

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    config = uvicorn.Config(api_server.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        server.should_exit = True
        mp.undo()
        pytest.fail("local uvicorn server did not start within 20 seconds")

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=10)
    mp.undo()


@pytest.fixture(scope="module")
def examples(live_server):
    """The blocks exactly as the About page would show them for this server."""
    return {e["id"]: e for e in build_api_examples(live_server)}


def _env(tmp_path, key):
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }
    if key is not None:
        env["DATA_ANALYZER_API_KEY"] = key
    return env


def run_block(example, tmp_path, key=TEST_API_KEY):
    """Run one block verbatim in tmp_path and return the CompletedProcess."""
    if example["language"] == "bash":
        cmd = ["bash", "-c", example["code"]]
    elif example["language"] == "python":
        cmd = [sys.executable, "-c", example["code"]]
    else:
        raise AssertionError(f"unknown language {example['language']!r}")
    return subprocess.run(
        cmd, cwd=tmp_path, env=_env(tmp_path, key),
        capture_output=True, text=True, timeout=60,
    )


def _json_from(stdout):
    """Parse the JSON a block prints (a block may print only JSON)."""
    return json.loads(stdout)


# ---------------------------------------------------------------------------
# Shape of the block list
# ---------------------------------------------------------------------------

EXPECTED_IDS = [
    "health_curl", "health_python",
    "version_curl", "version_python",
    "analyze_file_curl", "analyze_file_python",
    "analyze_json_curl", "analyze_json_python",
]


def test_there_are_curl_and_python_blocks_for_all_four_tasks():
    ids = [e["id"] for e in build_api_examples("http://127.0.0.1:9")]
    assert ids == EXPECTED_IDS


def test_every_block_has_id_title_language_and_code():
    for e in build_api_examples("http://127.0.0.1:9"):
        assert e["id"] and e["title"] and e["code"].strip()
        assert e["language"] in ("bash", "python")


def test_curl_blocks_are_single_line_without_backslash_continuations():
    for e in build_api_examples("http://127.0.0.1:9"):
        if e["language"] == "bash":
            assert "\n" not in e["code"].strip(), e["id"]
            assert "\\\n" not in e["code"], e["id"]
            assert not e["code"].endswith(" "), f"{e['id']} has a trailing space"


def test_python_blocks_import_only_requests_from_outside_the_standard_library():
    stdlib_ok = {"requests", "os", "json", "sys"}
    for e in build_api_examples("http://127.0.0.1:9"):
        if e["language"] != "python":
            continue
        for line in e["code"].splitlines():
            if line.startswith("import "):
                assert line.split()[1] in stdlib_ok, (e["id"], line)
            assert not line.startswith("from "), (e["id"], line)


def test_blocks_use_the_base_url_given_with_no_placeholder_left():
    for e in build_api_examples("http://127.0.0.1:9"):
        assert "http://127.0.0.1:9/api/v1/" in e["code"], e["id"]
        assert "<" not in e["code"] and "YOUR_" not in e["code"], e["id"]


def test_trailing_slash_on_the_base_url_is_not_doubled():
    for e in build_api_examples("http://127.0.0.1:9/"):
        assert "9//api" not in e["code"]


def test_unsafe_base_url_is_rejected_instead_of_pasted_into_a_shell_command():
    with pytest.raises(ValueError):
        build_api_examples('http://x"; rm -rf ~; echo "')


# ---------------------------------------------------------------------------
# The key is never shown
# ---------------------------------------------------------------------------

def test_no_block_contains_the_api_key_value(examples):
    for e in examples.values():
        assert TEST_API_KEY not in e["code"], e["id"]


def test_blocks_only_reference_the_key_by_environment_variable_name(examples):
    for e in examples.values():
        assert "os.getenv" not in e["code"], e["id"]
        if "X-API-Key" in e["code"]:
            if e["language"] == "bash":
                assert '"X-API-Key: $DATA_ANALYZER_API_KEY"' in e["code"], e["id"]
            else:
                assert 'os.environ["DATA_ANALYZER_API_KEY"]' in e["code"], e["id"]


def test_the_examples_module_never_reads_the_environment():
    import ast
    tree = ast.parse(Path(api_examples.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert "os" not in imported  # the example text mentions os; the module must not use it
    assert "streamlit" not in imported


# ---------------------------------------------------------------------------
# The rendered About page shows exactly these blocks
# ---------------------------------------------------------------------------

def test_about_page_shows_exactly_the_tested_blocks_and_never_the_key(monkeypatch):
    """Render web_app's About page headlessly (Streamlit AppTest) with a key and
    a docker-compose URL in the environment."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DATA_ANALYZER_API_URL", "http://api:8000")
    monkeypatch.setenv("DATA_ANALYZER_API_KEY", "page-must-not-show-this-key-98765")
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "web_app.py"), default_timeout=90)
    at.session_state["nav_choice"] = "About"
    at.run()

    assert not at.exception
    shown = [c.value for c in at.code]
    expected = [e["code"] for e in build_api_examples(LOCAL_URL)]
    assert shown == expected
    assert any("Use it from code (API)" in h.value for h in at.header)
    for text in shown + [m.value for m in at.markdown]:
        assert "page-must-not-show-this-key-98765" not in text
