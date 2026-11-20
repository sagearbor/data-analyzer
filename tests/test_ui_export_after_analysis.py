"""The Export controls appear in the SAME run that completes an analysis, default to a
visible download button, build the file from the current results, and never show an
old result after a failed analysis or a data change. Synthetic data only."""
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
    {"field_name": "pregnant", "data_type": "string", "description": "d", "branching_logic": "[gender]='Male'"},
    {"field_name": "smoker", "data_type": "string", "description": "d",
     "branching_logic": "[zz_marker] ~~ not a valid expression"},
    {"field_name": "age", "data_type": "int", "description": "d", "max_value": 100},
]
DATA = pd.DataFrame({"gender": ["Male", "Female"], "pregnant": ["1", ""], "smoker": ["n", "n"], "age": [500, 30]})


class FakeParser:
    deployment = "fake-deployment"

    def count_tokens(self, text):
        return 1

    def parse_dictionary(self, text, max_fields=500, deployment=None):
        return {"schema": {"age": {"type": "int", "max": 100}}, "fields": FIELDS,
                "metadata": {"chunks_processed": 1, "processing_time_seconds": 0.1, "mode": "fake"}}


class FakeResponse:
    def __init__(self, status=200):
        self.status_code = status
        self.text = "boom"

    def json(self):
        if self.status_code != 200:
            return {"detail": "synthetic failure"}
        return {"summary": {"total_rows": 2, "total_columns": 4, "issues_found": 1, "critical_issues": 1,
                            "warnings": 0, "data_types": {}, "completeness": 100.0, "issue_type_totals": {"range_violation": 1}},
                "issues": [{"type": "range_violation", "severity": "error", "column": "age", "row": 0,
                            "value": 500, "message": "Value 500 in column 'age' violates rule: max <= 100"}],
                "recommendations": [], "quality_checks": {}, "summary_stats": {}}


@pytest.fixture
def state(monkeypatch, tmp_path):
    monkeypatch.setenv("DICT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.delenv("CONTAINER_APP_NAME", raising=False)
    monkeypatch.setattr("src.llm_client.LLMDictionaryParser", FakeParser)
    box = {"status": 200}
    monkeypatch.setattr("requests.post", lambda *a, **k: FakeResponse(box["status"]))
    return box


def _loaded_app():
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    at.file_uploader(key="dict_uploader").upload("d.txt", b"anything", "application/octet-stream").run()
    at.file_uploader(key="data_uploader").upload("d.csv", DATA.to_csv(index=False).encode(), "text/csv").run()
    return at


def _export_visible(at):
    return any("Export" in m.value for m in at.markdown) and any(s.label == "Choose format:" for s in at.selectbox)


def test_export_appears_in_the_same_run_that_completes_the_analysis(state):
    at = _loaded_app()
    assert not _export_visible(at)
    at.button(key="run_analysis_main").click().run()
    assert not at.exception
    assert any("Analysis complete" in s.value for s in at.success)
    assert _export_visible(at)
    assert len(at.get("download_button")) == 1


def test_the_default_export_choice_leaves_a_download_button_visible_and_json_is_reachable(state):
    at = _loaded_app()
    at.button(key="run_analysis_main").click().run()
    box = [s for s in at.selectbox if s.label == "Choose format:"][0]
    assert box.value == "Excel with highlighting"
    assert at.get("download_button")[0].proto.label.endswith("Download Excel")
    box.select("JSON report").run()
    assert at.get("download_button")[0].proto.label.endswith("Download JSON")


def test_messages_and_warnings_still_appear_with_the_export(state):
    at = _loaded_app()
    at.button(key="run_analysis_main").click().run()
    text = " ".join(w.value for w in at.warning).replace("\\", "")
    assert "Partial result" in text and "1 of 2" in text
    assert any(e.label.replace("\\", "").startswith("Range Violation") for e in at.expander)


def test_a_failed_analysis_shows_its_error_and_no_export_for_the_previous_result(state):
    at = _loaded_app()
    at.button(key="run_analysis_main").click().run()
    assert _export_visible(at)
    state["status"] = 500
    at.button(key="run_analysis_main").click().run()
    assert any("Analysis failed" in e.value for e in at.error)
    assert not _export_visible(at)
    assert at.session_state["analysis_results"] is None
    assert not any(h.value == "Analysis Summary" for h in at.subheader)


def test_changing_the_data_clears_the_old_results_and_their_export(state):
    at = _loaded_app()
    at.button(key="run_analysis_main").click().run()
    assert _export_visible(at)
    other = pd.DataFrame({"gender": ["Male"], "pregnant": [""], "smoker": ["n"], "age": [5]})
    at.file_uploader(key="data_uploader").upload("e.csv", other.to_csv(index=False).encode(), "text/csv").run()
    assert at.session_state["analysis_results"] is None
    assert not _export_visible(at)
    assert any("results were cleared" in i.value for i in at.info)


def test_the_page_is_stable_across_reruns_without_a_change(state):
    at = _loaded_app()
    at.button(key="run_analysis_main").click().run()
    at.run()
    assert _export_visible(at) and at.session_state["analysis_results"] is not None


def _current_results():
    analyzer = web_app.DataQualityAnalyzer()
    import asyncio
    from unittest import mock

    with mock.patch("requests.post", lambda *a, **k: FakeResponse()):
        return asyncio.run(analyzer.analyze_data_quality(DATA, {"fields": FIELDS, "source": "t", "filename": "t"}))


def test_the_excel_export_for_the_current_results_has_data_issues_and_rule_coverage_sheets():
    results = _current_results()
    blob = web_app.export_to_excel_with_highlighting(
        DATA, results["issues"], results.get("rule_coverage"), results["summary"].get("column_types_not_checked"))
    wb = load_workbook(io.BytesIO(blob))
    assert {"Data", "Issues", "Rule coverage"} <= set(wb.sheetnames)
    flat = " ".join(str(c.value) for r in wb["Rule coverage"].iter_rows() for c in r)
    assert "PARTIAL - 1 of 2 rules could not be checked" in flat


def test_the_json_export_parses_and_carries_the_same_issue_total_as_the_page(state):
    at = _loaded_app()
    at.button(key="run_analysis_main").click().run()
    results = at.session_state["analysis_results"]
    parsed = json.loads(json.dumps(results, indent=2, default=str))
    shown = [m.value for m in at.metric if m.label == "Issues Found"]
    assert str(parsed["summary"]["issues_found"]) == shown[0]
    assert parsed["rule_coverage"]["rules_not_checked"] == 1
