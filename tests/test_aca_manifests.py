"""Deploy manifests must never carry secret values or wipe-able secrets lists.

Runs over deploy/aca/*.yaml, including the gitignored *.local.yaml copies when
they exist on the machine (those hold real infra IDs and are the real risk).
"""
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("check_aca_manifests", REPO / "scripts" / "check_aca_manifests.py")
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)

MANIFESTS = checker.manifest_files([])


@pytest.mark.unit
@pytest.mark.parametrize("path", MANIFESTS, ids=[p.name for p in MANIFESTS])
def test_repo_manifest_has_no_secret_values(path):
    assert checker.check_text(path.read_text(), path.name) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "snippet",
    [
        # placeholder value would overwrite the live secret on update --yaml
        "properties:\n  configuration:\n    secrets:\n      - name: k\n        value: \"<set-me>\"\n",
        # empty list would replace the app's secrets with nothing
        "properties:\n  configuration:\n    secrets: []\n",
        "properties:\n  configuration:\n    secrets:\n    registries: []\n",
        # name-only entry: not a Key Vault ref
        "properties:\n  configuration:\n    secrets:\n      - name: k\n",
        # Key Vault ref without identity
        "properties:\n  configuration:\n    secrets:\n      - name: k\n        keyVaultUrl: https://v.vault.azure.net/secrets/k\n",
    ],
)
def test_unsafe_snippets_are_refused(snippet):
    assert checker.check_text(snippet, "snippet")


@pytest.mark.unit
@pytest.mark.parametrize(
    "snippet",
    [
        # interim pattern: no secrets key at all; env uses secretRef
        "properties:\n  configuration:\n    ingress: {}\n  template:\n    containers:\n"
        "      - name: a\n        env:\n          - name: K\n            secretRef: k\n",
        # target pattern: Key Vault refs only
        "properties:\n  configuration:\n    secrets:\n      - name: k\n"
        "        keyVaultUrl: https://v.vault.azure.net/secrets/k\n        identity: /subscriptions/x/id\n",
        # a commented-out example block is fine
        "properties:\n  configuration:\n    # secrets:\n    #   - name: k\n    #     value: x\n    registries: []\n",
        # volume secrets (secretRef/path) are not app secrets
        "properties:\n  template:\n    volumes:\n      - name: v\n        storageType: Secret\n"
        "        secrets:\n          - secretRef: k\n            path: k.txt\n",
    ],
)
def test_safe_snippets_pass(snippet):
    assert checker.check_text(snippet, "snippet") == []
