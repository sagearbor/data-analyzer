"""ADP-82: conditional rules that could not be checked are shown as an explicit
PARTIAL result on the page and in the JSON and Excel exports. Synthetic data only."""
import asyncio
import io
import json
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import load_workbook
from streamlit.testing.v1 import AppTest

import web_app

APP = str(Path(__file__).resolve().parent.parent / "web_app.py")

FIELDS = [
    {"field_name": "gender", "data_type": "string", "description": "d"},
    # supported: shown only when gender = Male
    {"field_name": "pregnant", "data_type": "string", "description": "d", "branching_logic": "[gender]='Male'"},
    # unsupported: not a valid branching expression
    {"field_name": "smoker", "data_type": "string", "description": "d",
     "branching_logic": "[zz_marker] ~~ not a valid expression"},
]


class FakeParser:
    deployment = "fake-deployment"

    def count_tokens(self, text):
        return 1

    def parse_dictionary(self, text, max_fields=500, deployment=None):
        return {"schema": {}, "fields": FIELDS,
                "metadata": {"chunks_processed": 1, "processing_time_seconds": 0.1, "mode": "fake"}}


class FakeResponse:
    status_code = 200

    def json(self):
        return {"summary": {"total_rows": 2, "total_columns": 3, "issues_found": 0, "critical_issues": 0,
                            "warnings": 0, "data_types": {}, "completeness": 100.0},
                "issues": [], "recommendations": [], "quality_checks": {}, "summary_stats": {}}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("DICT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.delenv("CONTAINER_APP_NAME", raising=False)
    monkeypatch.setattr("src.llm_client.LLMDictionaryParser", FakeParser)
    monkeypatch.setattr("requests.post", lambda *a, **k: FakeResponse())


DATA = pd.DataFrame({"gender": ["Male", "Female"], "pregnant": ["1", ""], "smoker": ["n", "n"]})


def _analysis_results(dictionary):
    analyzer = web_app.DataQualityAnalyzer()
    return asyncio.run(analyzer.analyze_data_quality(DATA, dictionary))


def test_a_dictionary_with_a_supported_and_an_unsupported_rule_gives_an_explicit_partial_result():
    result = _analysis_results({"fields": FIELDS, "source": "t", "filename": "t.csv"})
    cov = result["rule_coverage"]
    assert (cov["rules_total"], cov["rules_evaluated"], cov["rules_not_checked"]) == (2, 1, 1)
    assert cov["complete"] is False
    assert cov["unsupported_rules"] == [{"rule_id": "smoker_show_if", "reason": "unsupported_condition"}]
    assert result["summary"]["rules_not_checked"] == 1


def test_a_dictionary_with_only_supported_rules_reports_complete_coverage():
    result = _analysis_results({"fields": FIELDS[:2], "source": "t", "filename": "t.csv"})
    assert result["rule_coverage"]["complete"] is True
    assert result["summary"]["rules_not_checked"] == 0


def test_the_page_says_how_many_rules_could_not_be_checked_and_that_the_result_is_partial():
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    at.file_uploader(key="dict_uploader").upload("d.txt", b"anything", "application/octet-stream").run()
    at.file_uploader(key="data_uploader").upload("d.csv", DATA.to_csv(index=False).encode(), "text/csv").run()
    at.button(key="run_analysis_main").click().run()
    assert not at.exception
    warnings = [w.value for w in at.warning]
    partial = [w for w in warnings if "Partial result" in w]
    assert partial, warnings
    assert "1 of 2" in partial[0].replace("\\", "")
    # the rule id is dictionary-derived text: it is shown escaped
    assert any("smoker\\_show\\_if" in m.value for m in at.markdown)
    results = at.session_state["analysis_results"]
    assert results["summary"]["rules_not_checked"] == 1
    # the JSON export serialises exactly this dict
    assert json.loads(json.dumps(results, default=str))["rule_coverage"]["rules_not_checked"] == 1


def test_the_excel_export_has_a_rule_coverage_sheet_that_says_partial():
    cov = _analysis_results({"fields": FIELDS, "source": "t", "filename": "t.csv"})["rule_coverage"]
    wb = load_workbook(io.BytesIO(web_app.export_to_excel_with_highlighting(DATA, [], cov)))
    rows = [[c.value for c in r] for r in wb["Rule coverage"].iter_rows(values_only=False)]
    flat = " | ".join(str(v) for r in rows for v in r)
    assert "PARTIAL - 1 of 2 rules could not be checked" in flat
    assert "smoker_show_if" in flat


def test_a_hostile_rule_id_in_the_excel_coverage_sheet_is_stored_as_text_not_a_formula():
    cov = {"rules_total": 1, "rules_evaluated": 0, "rules_not_checked": 1, "complete": False,
           "unsupported_rules": [{"rule_id": "=HYPERLINK(\"http://x.invalid\")", "reason": "unsupported_condition"}],
           "failed_rules": []}
    wb = load_workbook(io.BytesIO(web_app.export_to_excel_with_highlighting(DATA, [], cov)))
    cell = next(c for r in wb["Rule coverage"].iter_rows() for c in r if str(c.value).startswith("=HYPERLINK"))
    assert cell.data_type == "s"
