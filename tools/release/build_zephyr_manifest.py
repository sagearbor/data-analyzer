#!/usr/bin/env python3
"""Assemble tests/zephyr/release-1.0-manifest.json from the per-file fragments.

Usage: build_zephyr_manifest.py <junit.xml> [--commit SHA]
- Keeps case keys already written into an existing manifest (so a re-run
  updates the same Zephyr cases instead of creating new ones).
- Fails if the JUnit file holds a test with no description, and lists
  descriptions whose test no longer exists (those are dropped).
"""
import glob, json, os, subprocess, sys, xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "tests/zephyr/release-1.0-manifest.json")
TRACK = json.load(open(os.path.join(ROOT, "docs/release-1.0/tracking.json")))
ISSUES = {v["key"]: int(v["id"]) for v in TRACK["issues"].values()}
ISSUES[TRACK["epic"]] = int(TRACK["epic_id"])
KEY = {slug: v["key"] for slug, v in TRACK["issues"].items()}
# Which story each test file gives evidence for (default: the site-test story).
FILE_ISSUE = {
    "test_api_programmatic": KEY["api-json"],
    "test_about_api_examples": KEY["about-api"],
    "test_llm_deployment_selection": KEY["model-choice"],
}
SEC = KEY["security-review"]

def _environment():
    """Describe the interpreter running this script; run it with the same
    Python that produced the JUnit file so the statement is true."""
    import platform
    from importlib.metadata import version
    return "Developer workstation (%s, Python %s, Streamlit %s, Pillow %s)" % (
        platform.system().replace("Darwin", "macOS"), platform.python_version(),
        version("streamlit"), version("pillow"))

ENVIRONMENT = _environment()

def main():
    junit = sys.argv[1]
    commit = sys.argv[sys.argv.index("--commit") + 1] if "--commit" in sys.argv else \
        subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"]).decode().strip()
    ran = []
    for tc in ET.parse(junit).iter("testcase"):
        i = tc.get("classname") + "." + tc.get("name").split("[")[0]
        if i not in ran:
            ran.append(i)
    old = {}
    if os.path.exists(OUT):
        old = {t["id"]: t for t in json.load(open(OUT))["tests"]}
    desc = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "tests/zephyr/fragments/*.json"))):
        for t in json.load(open(f)):
            desc[t["id"]] = t
    missing = [i for i in ran if i not in desc]
    stale = sorted(set(desc) - set(ran))
    tests = []
    for i in ran:
        if i not in desc:
            continue
        t = dict(desc[i])
        stem = i.split(".")[1]
        t["issue"] = SEC if stem.startswith("test_sec_") else FILE_ISSUE.get(stem, KEY["site-test"])
        for k, v in old.get(i, {}).items():      # keep Zephyr keys written back by `zs record`
            if k not in t:
                t[k] = v
        tests.append(t)
    names = [t["name"] for t in tests]
    dup = sorted({n for n in names if names.count(n) > 1})
    manifest = {
        "project": "ADP", "folder": "Data Analyzer 1.0",
        "source_repo": "sagearbor/data-analyzer", "branch": subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip(), "commit": commit,
        "run_command": "python -m pytest tests -p no:cacheprovider --junitxml=results.xml",
        "environment": ENVIRONMENT + ", offline: no network and no AI model (AI calls are replaced by canned answers), made-up data",
        "issue_ids": ISSUES, "cycle_issues": [TRACK["epic"], KEY["site-test"]],
        "tests": tests,
    }
    json.dump(manifest, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"ran {len(ran)} distinct tests; described {len(tests)}; missing {len(missing)}; stale descriptions dropped {len(stale)}; duplicate names {len(dup)}")
    for m in missing: print("  MISSING", m)
    for d in dup: print("  DUPLICATE NAME", d)
    sys.exit(1 if (missing or dup) else 0)

if __name__ == "__main__":
    main()
