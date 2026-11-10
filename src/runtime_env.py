"""
Runtime-environment detection and fail-fast configuration checks.

One place decides "is this a deployed (non-local) environment?", so the API's
API-key guard and the UI's API-connection guard can't drift apart.

Deployed means EITHER:
  * running on Azure Container Apps - ACA injects CONTAINER_APP_NAME,
    CONTAINER_APP_REVISION, CONTAINER_APP_HOSTNAME, CONTAINER_APP_ENV_DNS_SUFFIX
    into every replica; ANY of them counts, so this holds even if a manifest
    forgets (or mis-sets) APP_ENV; OR
  * APP_ENV is anything other than a known local value (prod, staging, ...).

Local values: dev, development, local, test (APP_ENV unset counts as dev).

Run as a module for the UI container's startup check (entrypoint.sh):
    python -m src.runtime_env --check-ui
Exits 1 with one ERROR line per problem, so a misconfigured revision never
becomes ready instead of serving a page that can't analyze anything.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import List, Mapping, Optional
from urllib.parse import urlparse

logger = logging.getLogger("data_analyzer.config")

LOCAL_APP_ENVS = frozenset({"dev", "development", "local", "test"})

# Local-dev default for the UI -> API connection (api_server.py's default port).
LOCAL_API_URL = "http://localhost:8000"

# Env vars Azure Container Apps injects into every replica (any one = on ACA).
ACA_MARKER_VARS = ("CONTAINER_APP_NAME", "CONTAINER_APP_REVISION",
                   "CONTAINER_APP_HOSTNAME", "CONTAINER_APP_ENV_DNS_SUFFIX")

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


def is_deployed_env(environ: Optional[Mapping[str, str]] = None) -> bool:
    """True when running anywhere other than a developer's machine / test run."""
    env = os.environ if environ is None else environ
    if any((env.get(name) or "").strip() for name in ACA_MARKER_VARS):
        return True
    app_env = (env.get("APP_ENV") or "dev").strip().lower()
    return app_env not in LOCAL_APP_ENVS


class ApiConfigError(RuntimeError):
    """The UI is not configured to reach the data-analyzer API."""


def _is_loopback_url(url: str) -> bool:
    try:
        host = (urlparse(url if "://" in url else f"http://{url}").hostname or "").lower()
    except ValueError:
        return False
    return host in _LOOPBACK_HOSTS or host.startswith("127.")


def resolve_api_base_url(environ: Optional[Mapping[str, str]] = None) -> str:
    """Return the data-analyzer API base URL the UI should call.

    Locally an unset DATA_ANALYZER_API_URL falls back to LOCAL_API_URL. In a
    deployed environment there is NO fallback - localhost is never right inside
    a container app - so ApiConfigError is raised instead.
    """
    env = os.environ if environ is None else environ
    url = (env.get("DATA_ANALYZER_API_URL") or "").strip().rstrip("/")
    deployed = is_deployed_env(env)
    if url:
        if deployed and _is_loopback_url(url):
            # Inside a container app localhost is the UI's own replica, never
            # the API: an explicit localhost URL is as broken as a missing one.
            raise ApiConfigError(
                "Configuration error: DATA_ANALYZER_API_URL points at localhost in a "
                "deployed environment, so the UI cannot reach the analysis service. "
                "Ask the app owner to set it to the API app's https URL."
            )
        return url
    if deployed:
        raise ApiConfigError(
            "Configuration error: DATA_ANALYZER_API_URL is not set for this deployment, "
            "so the UI cannot reach the analysis service. Ask the app owner to set it."
        )
    return LOCAL_API_URL


def ui_config_problems(environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """Problems that make the deployed UI unable to run an analysis.

    Empty list = OK. Always empty locally (the localhost default applies and
    the local API may run without a key).
    """
    env = os.environ if environ is None else environ
    if not is_deployed_env(env):
        return []
    problems = []
    url = (env.get("DATA_ANALYZER_API_URL") or "").strip()
    if not url:
        problems.append("DATA_ANALYZER_API_URL is not set (UI cannot reach the API)")
    elif _is_loopback_url(url):
        problems.append("DATA_ANALYZER_API_URL points at localhost (UI cannot reach the API)")
    if not (env.get("DATA_ANALYZER_API_KEY") or "").strip():
        problems.append("DATA_ANALYZER_API_KEY is not set (the API will answer 401)")
    return problems


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv != ["--check-ui"]:
        print("usage: python -m src.runtime_env --check-ui", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    problems = ui_config_problems()
    for problem in problems:
        logger.error("ui_config_check_failed problem=%r", problem)
    if problems:
        logger.error("Refusing to start the UI: fix the container app's env vars/secrets.")
        return 1
    logger.info("ui_config_check_ok deployed=%s", is_deployed_env())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
