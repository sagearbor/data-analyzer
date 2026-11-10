#!/usr/bin/env python3
"""Refuse Azure Container Apps manifests that could overwrite, wipe or leak a secret.

Stdlib only (runs in CI and in a bare venv). This is the repo-local mirror of
the shared deploy guard (`~/dcri/aca-guard.sh scan`), so a bad manifest fails
`pytest` before anyone can deploy it.

Why each rule exists (`az containerapp update --yaml` sends a JSON Merge PATCH;
`create --yaml` sends a full PUT):
  * a secret with `value:` -- a real value leaks into the file; a placeholder
    OVERWRITES the live secret on the next deploy;
  * `secrets: []` / an empty `secrets:` key -- replaces the app's secret list
    with nothing;
  * a secret entry without `keyVaultUrl` -- only Key Vault references
    (name + keyVaultUrl + identity) are allowed inside a manifest.
Leaving the `secrets:` key out entirely is the safe interim pattern: the PATCH
then leaves the app's secrets untouched.

Usage:  python scripts/check_aca_manifests.py [FILE_OR_DIR ...]
        (default: deploy/aca, including gitignored *.local.yaml copies)
Exit 1 and a list of problems (never values) if anything is unsafe.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KEY_RE = re.compile(r"^(?P<indent>\s*)(?:-\s+)?(?P<key>[A-Za-z_][\w-]*)\s*:\s*(?P<rest>.*)$")


def _strip_comment(line: str) -> str:
    # good enough for manifests: a ' #' outside quotes starts a comment
    out, quote = [], None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            break
        out.append(ch)
    return "".join(out).rstrip()


def check_text(text: str, label: str) -> list[str]:
    problems: list[str] = []
    lines = [_strip_comment(raw) for raw in text.splitlines()]
    i = 0
    while i < len(lines):
        m = KEY_RE.match(lines[i])
        # the app-level list is `configuration.secrets` (volumes also use a
        # `secrets:` key, but its items are secretRef/path, never values)
        if not m or m.group("key") != "secrets":
            i += 1
            continue
        indent = len(m.group("indent"))
        rest = m.group("rest").strip()
        block: list[str] = []
        j = i + 1
        while j < len(lines):
            ln = lines[j]
            if ln.strip() == "":
                j += 1
                continue
            lead = len(ln) - len(ln.lstrip())
            # list items may sit at the same indent as the key ("secrets:\n- name")
            if lead < indent or (lead == indent and not ln.lstrip().startswith("-")):
                break
            block.append(ln)
            j += 1
        where = f"{label}:{i + 1}"
        if rest in ("[]", "~", "null") or (rest == "" and not block):
            problems.append(f"{where}: empty `secrets:` list would wipe the app's secrets; delete the key")
        items: list[dict[str, str]] = []
        for ln in block:
            km = KEY_RE.match(ln)
            if not km:
                continue
            if ln.lstrip().startswith("-"):
                items.append({})
            if items:
                items[-1][km.group("key")] = km.group("rest").strip()
        for item in items:
            name = item.get("name", "?").strip("\"'")
            if "secretRef" in item:  # a volume's secrets list, not app secrets
                continue
            if "value" in item:
                problems.append(f"{where}: secret `{name}` has `value:`; secret values never go in a manifest")
            if "keyVaultUrl" not in item:
                problems.append(f"{where}: secret `{name}` is not a Key Vault ref (name + keyVaultUrl + identity)")
            elif "identity" not in item:
                problems.append(f"{where}: secret `{name}` has keyVaultUrl but no identity")
        i = j
    return problems


def manifest_files(targets: list[str]) -> list[Path]:
    paths = [Path(t) for t in targets] or [REPO / "deploy" / "aca"]
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += sorted(list(p.glob("*.yaml")) + list(p.glob("*.yml")))
        elif p.exists():
            files.append(p)
    return files


def main(argv: list[str]) -> int:
    files = manifest_files(argv)
    problems: list[str] = []
    for f in files:
        problems += check_text(f.read_text(), str(f.relative_to(REPO)) if f.is_relative_to(REPO) else str(f))
    if problems:
        print("ACA manifest check FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"ACA manifest check OK ({len(files)} file(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
