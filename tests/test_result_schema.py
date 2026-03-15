"""Tests for result_schema.py - result.json parsing and validation."""

import json
from pathlib import Path

import pytest

from codefleet.models import ResultStatus, TestStatus
from codefleet.result_schema import (
    ResultValidationError,
    parse_result_file,
    validate_result_data,
)


class TestParseResultFile:
    def test_valid_complete_result(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text(
            json.dumps(
                {
                    "summary": "Implemented feature X",
                    "files_changed": ["src/x.py", "tests/test_x.py"],
                    "tests": [
                        {
                            "command": "pytest tests/test_x.py",
                            "status": "passed",
                            "details": "5 passed",
                        }
                    ],
                    "commits": ["abc123"],
                    "next_steps": ["Add docs"],
                    "status": "completed",
                }
            )
        )
        result = parse_result_file(result_file)
        assert result.summary == "Implemented feature X"
        assert result.files_changed == ["src/x.py", "tests/test_x.py"]
        assert len(result.tests) == 1
        assert result.tests[0].status == TestStatus.PASSED
        assert result.commits == ["abc123"]
        assert result.status == ResultStatus.COMPLETED

    def test_valid_minimal_result(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text(
            json.dumps({"summary": "Done", "status": "completed"})
        )
        result = parse_result_file(result_file)
        assert result.summary == "Done"
        assert result.files_changed == []
        assert result.tests == []

    def test_valid_blocked_result(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text(
            json.dumps(
                {
                    "summary": "Cannot proceed without dependency",
                    "status": "blocked",
                    "next_steps": ["Install package X"],
                }
            )
        )
        result = parse_result_file(result_file)
        assert result.status == ResultStatus.BLOCKED
        assert result.next_steps == ["Install package X"]

    def test_missing_file(self, tmp_path):
        result_file = tmp_path / "nonexistent.json"
        with pytest.raises(ResultValidationError, match="not found"):
            parse_result_file(result_file)

    def test_empty_file(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text("")
        with pytest.raises(ResultValidationError, match="empty"):
            parse_result_file(result_file)

    def test_whitespace_only_file(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text("   \n\n  ")
        with pytest.raises(ResultValidationError, match="empty"):
            parse_result_file(result_file)

    def test_invalid_json(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text("{not valid json}")
        with pytest.raises(ResultValidationError, match="Invalid JSON"):
            parse_result_file(result_file)

    def test_json_array_not_object(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text('[{"summary": "test"}]')
        with pytest.raises(ResultValidationError, match="must be an object"):
            parse_result_file(result_file)

    def test_missing_required_field(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps({"summary": "test"}))  # missing status
        with pytest.raises(ResultValidationError, match="validation failed"):
            parse_result_file(result_file)

    def test_invalid_status_value(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text(
            json.dumps({"summary": "test", "status": "unknown"})
        )
        with pytest.raises(ResultValidationError, match="validation failed"):
            parse_result_file(result_file)

    def test_invalid_test_status(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text(
            json.dumps(
                {
                    "summary": "test",
                    "status": "completed",
                    "tests": [
                        {"command": "pytest", "status": "broken"}
                    ],
                }
            )
        )
        with pytest.raises(ResultValidationError, match="validation failed"):
            parse_result_file(result_file)

    def test_all_test_statuses(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text(
            json.dumps(
                {
                    "summary": "test",
                    "status": "completed",
                    "tests": [
                        {"command": "pytest test_a.py", "status": "passed"},
                        {"command": "pytest test_b.py", "status": "failed", "details": "1 failed"},
                        {"command": "pytest test_c.py", "status": "not_run"},
                    ],
                }
            )
        )
        result = parse_result_file(result_file)
        assert len(result.tests) == 3
        assert result.tests[0].status == TestStatus.PASSED
        assert result.tests[1].status == TestStatus.FAILED
        assert result.tests[2].status == TestStatus.NOT_RUN


class TestValidateResultData:
    def test_valid_data(self):
        data = {
            "summary": "Done",
            "files_changed": ["a.py"],
            "status": "completed",
        }
        result = validate_result_data(data)
        assert result.summary == "Done"

    def test_invalid_data(self):
        with pytest.raises(ResultValidationError):
            validate_result_data({"bad": "data"})

    def test_extra_fields_ignored(self):
        data = {
            "summary": "Done",
            "status": "completed",
            "extra_field": "should not cause error",
        }
        result = validate_result_data(data)
        assert result.summary == "Done"
