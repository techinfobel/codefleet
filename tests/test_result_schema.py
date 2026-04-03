"""Tests for result parsing and validation (parse_result_file in models.py)."""

import json
from pathlib import Path

import pytest

from codefleet.models import ResultStatus, WorkerResult, parse_result_file


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
                            "command": "uv run pytest",
                            "status": "passed",
                            "details": "12 passed",
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
        assert result.tests[0].command == "uv run pytest"
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
        with pytest.raises(FileNotFoundError, match="not found"):
            parse_result_file(result_file)

    def test_empty_file(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text("")
        with pytest.raises(ValueError, match="empty"):
            parse_result_file(result_file)

    def test_whitespace_only_file(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text("   \n\n  ")
        with pytest.raises(ValueError, match="empty"):
            parse_result_file(result_file)

    def test_invalid_json(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text("{not valid json}")
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_result_file(result_file)

    def test_json_array_not_object(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text('[{"summary": "test"}]')
        with pytest.raises(ValueError, match="must be an object"):
            parse_result_file(result_file)

    def test_missing_required_field(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps({"summary": "test"}))  # missing status
        with pytest.raises(Exception):
            parse_result_file(result_file)

    def test_invalid_status_value(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text(
            json.dumps({"summary": "test", "status": "unknown"})
        )
        with pytest.raises(Exception):
            parse_result_file(result_file)


class TestValidateResultData:
    def test_valid_data(self):
        data = {
            "summary": "Done",
            "files_changed": ["a.py"],
            "status": "completed",
        }
        result = WorkerResult.model_validate(data)
        assert result.summary == "Done"

    def test_invalid_data(self):
        with pytest.raises((ValueError, Exception)):
            WorkerResult.model_validate({"bad": "data"})

    def test_extra_fields_ignored(self):
        data = {
            "summary": "Done",
            "status": "completed",
            "extra_field": "should not cause error",
        }
        result = WorkerResult.model_validate(data)
        assert result.summary == "Done"
