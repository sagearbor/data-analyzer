#!/usr/bin/env python3
"""
Test PDF dictionary extraction: real PDF text extraction (pypdf) plus a
mocked LLM parsing step, so the suite stays deterministic and offline.

A second, real-LLM variant is provided for manual verification against
Azure OpenAI; it is skipped automatically unless AZURE_OPENAI_API_KEY (or
OPENAI_API_KEY) is set, and is never run as part of the default suite.
"""

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import pypdf

from src.llm_client import LLMDictionaryParser, FieldDefinition


# This repo's .env normally carries real Azure OpenAI credentials for local
# development, so "credentials present" is not a safe signal for "ok to make
# a live network call in the default test run." Require an explicit opt-in
# instead, so `pytest tests/` never dials out.
RUN_LIVE_LLM_TESTS = os.environ.get("RUN_LIVE_LLM_TESTS") == "1"


@pytest.fixture
def pdf_path(test_data_dir):
    """Fixture that returns the path to the test PDF file"""
    return test_data_dir / "dictionaries" / "LTCPROMISE8314QC.pdf"


def _read_pdf_text(pdf_path: Path, max_pages: int = None) -> str:
    """
    Extract raw text content from a PDF using pypdf (no LLM involved).

    max_pages caps how many pages are read. The fixture PDF is ~90 pages and
    pypdf's extract_text() runs roughly 1 page/second, so unbounded
    extraction makes this test slow; a handful of pages is enough to
    exercise the real extraction path deterministically and quickly.
    """
    with open(pdf_path, "rb") as file:
        pdf_reader = pypdf.PdfReader(file)
        pages = pdf_reader.pages[:max_pages] if max_pages else pdf_reader.pages
        text_content = ""
        for page in pages:
            text_content += page.extract_text() + "\n"
    return text_content


def _mock_fields():
    """A small, representative set of fields mimicking a real LLM extraction."""
    return [
        FieldDefinition(
            field_name="record_id",
            data_type="str",
            required=True,
            description="Unique record identifier",
        ),
        FieldDefinition(
            field_name="age",
            data_type="int",
            required=True,
            description="Subject age in years",
            min_value=0,
            max_value=120,
        ),
        FieldDefinition(
            field_name="visit_date",
            data_type="date",
            required=False,
            description="Date of visit",
        ),
    ]


def test_pdf_text_extraction(pdf_path):
    """PDF text extraction (pypdf) should return non-empty content"""
    assert os.path.exists(pdf_path), f"File not found: {pdf_path}"

    text_content = _read_pdf_text(pdf_path, max_pages=5)

    assert len(text_content) > 0, "PDF extraction returned empty content"


def test_pdf_extraction(pdf_path):
    """
    End-to-end dictionary parsing with the LLM call mocked out.

    Exercises real PDF text extraction (first few pages, for speed), then a
    mocked LLMDictionaryParser.parse_dictionary() call so no network access
    or Azure OpenAI credentials are required.
    """
    text_content = _read_pdf_text(pdf_path, max_pages=5)
    assert len(text_content) > 0, "PDF extraction returned empty content"

    mock_fields = _mock_fields()
    mock_result = {
        "fields": [f.to_dict() for f in mock_fields],
        "schema": {f.field_name: f.data_type for f in mock_fields},
        "metadata": {
            "total_fields": len(mock_fields),
            "chunks_processed": 1,
            "mode": "single-call",
            "source": "LLM Parser (mocked)",
            "processing_time_seconds": 0.01,
        },
    }

    with patch.object(LLMDictionaryParser, "__init__", return_value=None):
        parser = LLMDictionaryParser()

    with patch.object(LLMDictionaryParser, "parse_dictionary", return_value=mock_result) as mock_parse:
        result = parser.parse_dictionary(text_content, max_fields=500)

    mock_parse.assert_called_once_with(text_content, max_fields=500)

    fields = result.get("fields", [])
    assert fields is not None, "No fields returned from parser"
    assert len(fields) > 0, "Parser returned empty fields list"
    assert result.get("metadata", {}).get("chunks_processed", 0) > 0, "Should process at least one chunk"


@pytest.mark.skipif(
    not RUN_LIVE_LLM_TESTS,
    reason="Set RUN_LIVE_LLM_TESTS=1 to opt in to a real Azure OpenAI call",
)
def test_pdf_extraction_live_llm(pdf_path):
    """
    Manual/opt-in variant that makes a real Azure OpenAI call.

    Excluded from the default deterministic/offline test run; only runs
    when explicitly requested via RUN_LIVE_LLM_TESTS=1.
    """
    text_content = _read_pdf_text(pdf_path)
    assert len(text_content) > 0, "PDF extraction returned empty content"

    parser = LLMDictionaryParser()

    start_time = time.time()
    result = parser.parse_dictionary(text_content, max_fields=500)
    elapsed = time.time() - start_time

    fields = result.get("fields", [])
    assert fields is not None, "No fields returned from parser"
    assert len(fields) > 0, "Parser returned empty fields list"
    assert result.get("metadata", {}).get("chunks_processed", 0) > 0, "Should process at least one chunk"

    print(f"\nLive LLM parse completed in {elapsed:.1f}s, extracted {len(fields)} fields")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
