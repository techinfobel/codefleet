import json
from pathlib import Path

from .models import WorkerResult


class ResultValidationError(Exception):
    pass


def _validate_result(data: object) -> WorkerResult:
    if not isinstance(data, dict):
        raise ResultValidationError("Result JSON must be an object")
    try:
        return WorkerResult.model_validate(data)
    except Exception as e:
        raise ResultValidationError(f"Result schema validation failed: {e}")


def parse_result_file(path: Path) -> WorkerResult:
    """Parse and validate a result.json file."""
    if not path.exists():
        raise ResultValidationError(f"Result file not found: {path}")

    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        raise ResultValidationError(f"Result file is empty: {path}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ResultValidationError(f"Invalid JSON in result file: {e}")

    return _validate_result(data)


def validate_result_data(data: dict) -> WorkerResult:
    """Validate a result dict against the schema."""
    return _validate_result(data)
