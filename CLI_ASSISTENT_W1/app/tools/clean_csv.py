"""Tool for cleaning, analyzing, and visualizing CSV data in the terminal."""

from __future__ import annotations

import contextlib
import csv
import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.llm.base import ToolDefinition


class CleanCSVInput(BaseModel):
    """Input parameters for the clean_csv tool."""

    model_config = ConfigDict(extra="forbid")

    file_path: str = Field(
        ...,
        description="Path to the CSV file to clean/analyze (e.g. 'workspace/sample_data.csv').",
    )
    drop_duplicates: bool = Field(
        default=True,
        description="Whether to drop duplicate rows.",
    )
    duplicate_subset: list[str] | str | None = Field(
        default=None,
        description=(
            "Optional column name or list of column names to consider when identifying "
            "duplicate rows. Defaults to all columns."
        ),
    )

    @field_validator("duplicate_subset", mode="before")
    @classmethod
    def _normalize_subset(cls, value: object) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            # Support single string or comma-separated column names
            return [col.strip() for col in value.split(",") if col.strip()]
        if isinstance(value, list):
            return [str(v).strip() for v in value]
        return None

    empty_row_strategy: Literal["drop_any_empty", "drop_all_empty", "fill_defaults", "keep"] = (
        Field(
            default="drop_any_empty",
            description=(
                "How to handle empty cells or rows: 'drop_any_empty', 'drop_all_empty', "
                "'fill_defaults', or 'keep'."
            ),
        )
    )
    fill_values: dict[str, str] | None = Field(
        default=None,
        description="Optional mapping of column names to custom replacement strings.",
    )
    visualize_column: str | None = Field(
        default=None,
        description="Name of a column to generate an ASCII graphical chart or histogram for.",
    )
    detect_outliers_column: str | None = Field(
        default=None,
        description="Name of a numerical column to detect outliers in using the IQR method.",
    )
    normalize_column_names: bool = Field(
        default=True,
        description="Whether to strip whitespace and normalize column headers to snake_case.",
    )
    output_file_path: str | None = Field(
        default=None,
        description="Optional path where the cleaned CSV will be saved.",
    )


class CleanCSVSummary(BaseModel):
    initial_rows: int
    cleaned_rows: int
    columns: list[str]
    duplicates_removed: int
    empty_rows_removed: int
    missing_values_per_column: dict[str, int]


class CleanCSVResult(BaseModel):
    status: Literal["success", "error"]
    summary: CleanCSVSummary
    column_types: dict[str, str]
    statistics: dict[str, Any]
    outliers: dict[str, Any] | None = None
    visualization: str | None = None
    saved_to: str | None = None
    preview: list[dict[str, Any]]


def _snake_case(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", "", text.strip())
    return re.sub(r"[\s\-]+", "_", cleaned).lower()


def _infer_type(values: Sequence[str]) -> str:
    non_empty = [v.strip() for v in values if v.strip()]
    if not non_empty:
        return "empty"

    is_int = True
    is_float = True
    for v in non_empty:
        try:
            int(v)
        except ValueError:
            is_int = False
            try:
                float(v)
            except ValueError:
                is_float = False
                break

    if is_int:
        return "integer"
    if is_float:
        return "float"
    if all(v.lower() in {"true", "false", "yes", "no", "0", "1"} for v in non_empty):
        return "boolean"
    if any(re.match(r"^\d{4}[-/]\d{2}[-/]\d{2}", v) for v in non_empty):
        return "date"
    return "text"


def _calculate_numeric_stats(numbers: list[float]) -> dict[str, float]:
    if not numbers:
        return {}
    numbers_sorted = sorted(numbers)
    n = len(numbers_sorted)
    total = sum(numbers_sorted)
    mean = total / n
    median = (
        numbers_sorted[n // 2]
        if n % 2 != 0
        else (numbers_sorted[n // 2 - 1] + numbers_sorted[n // 2]) / 2.0
    )
    variance = sum((x - mean) ** 2 for x in numbers_sorted) / max(1, n - 1)
    std_dev = math.sqrt(variance)
    return {
        "count": float(n),
        "min": round(numbers_sorted[0], 2),
        "max": round(numbers_sorted[-1], 2),
        "mean": round(mean, 2),
        "median": round(median, 2),
        "std_dev": round(std_dev, 2),
    }


def _detect_iqr_outliers(numbers: list[float]) -> dict[str, Any]:
    if len(numbers) < 4:
        return {"outliers_detected": 0, "message": "Not enough data points for IQR"}
    sorted_nums = sorted(numbers)
    n = len(sorted_nums)
    q1 = sorted_nums[n // 4]
    q3 = sorted_nums[(3 * n) // 4]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = [x for x in sorted_nums if x < lower_bound or x > upper_bound]
    return {
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "outlier_count": len(outliers),
        "outlier_values": [round(x, 2) for x in outliers[:10]],
    }


def _render_ascii_bar(ratio: float, max_width: int = 25) -> str:
    filled_len = int(round(ratio * max_width))
    return "█" * filled_len + "░" * (max_width - filled_len)


def _render_categorical_chart(column_name: str, values: list[str]) -> str:
    counts = Counter(values).most_common(8)
    total = len(values)
    if total == 0:
        return f"No data to visualize for '{column_name}'."

    max_label_len = max((len(str(val)) for val, _ in counts), default=5)
    lines = [f"📊 Category Distribution: {column_name} (Total: {total})", "─" * 55]
    for val, count in counts:
        ratio = count / total
        pct = ratio * 100
        label = str(val).ljust(max_label_len)
        bar = _render_ascii_bar(ratio, max_width=20)
        lines.append(f"  {label} │ {bar} {count:>3} ({pct:>5.1f}%)")
    return "\n".join(lines)


def _render_numerical_histogram(column_name: str, numbers: list[float]) -> str:
    if not numbers:
        return f"No numeric data to visualize for '{column_name}'."
    min_val = min(numbers)
    max_val = max(numbers)
    if min_val == max_val:
        return f"All values are {min_val} in column '{column_name}'."

    bin_count = min(6, len(set(numbers)))
    bin_width = (max_val - min_val) / bin_count
    bins: list[int] = [0] * bin_count

    for val in numbers:
        idx = int((val - min_val) / bin_width)
        if idx >= bin_count:
            idx = bin_count - 1
        bins[idx] += 1

    total = len(numbers)
    lines = [
        f"📊 Histogram: {column_name} (Min: {min_val:.1f}, Max: {max_val:.1f}, N: {total})",
        "─" * 60,
    ]
    for i, count in enumerate(bins):
        low = min_val + i * bin_width
        high = min_val + (i + 1) * bin_width
        ratio = count / total
        pct = ratio * 100
        bar = _render_ascii_bar(ratio, max_width=20)
        range_label = f"[{low:>7.1f} - {high:>7.1f}]"
        lines.append(f"  {range_label} │ {bar} {count:>3} ({pct:>5.1f}%)")
    return "\n".join(lines)


class CleanCSVTool:
    """Tool that cleans CSV files, computes statistics, detects outliers, and visualizes columns."""

    def __init__(self, workspace_dir: str = "workspace") -> None:
        self._workspace_dir = Path(workspace_dir)

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="clean_csv",
            description=(
                "Clean and profile CSV datasets. Handles duplicate entries, removes or "
                "imputes empty/missing rows, normalizes headers, computes statistics and "
                "outlier detection, and renders graphical ASCII charts/histograms."
            ),
            parameters=CleanCSVInput.model_json_schema(),
        )

    def execute(self, arguments_json: str) -> str:
        try:
            raw = json.loads(arguments_json)
        except Exception as exc:
            return json.dumps(
                {
                    "status": "error",
                    "error_type": "JSONDecodeError",
                    "message": f"Malformed JSON arguments: {exc}",
                }
            )

        if isinstance(raw, dict):
            if "parameters" in raw and isinstance(raw["parameters"], dict):
                raw = raw["parameters"]
            elif "arguments" in raw and isinstance(raw["arguments"], dict):
                raw = raw["arguments"]

        try:
            params = CleanCSVInput.model_validate(raw)
        except ValidationError as exc:
            return json.dumps(
                {
                    "status": "error",
                    "error_type": "ValidationError",
                    "message": f"Invalid arguments for clean_csv: {exc}",
                }
            )

        path = Path(params.file_path)
        if not path.is_file():
            # Try searching in workspace directory if path is relative
            candidate = self._workspace_dir / path.name
            if candidate.is_file():
                path = candidate
            else:
                return json.dumps(
                    {
                        "status": "error",
                        "error_type": "FileNotFoundError",
                        "message": f"CSV file not found at '{params.file_path}'.",
                    }
                )

        try:
            with open(path, encoding="utf-8-sig", errors="replace") as f:
                reader = csv.reader(f)
                rows = list(reader)
        except Exception as exc:
            return json.dumps(
                {
                    "status": "error",
                    "error_type": "FileReadError",
                    "message": f"Failed reading CSV file: {exc}",
                }
            )

        if not rows:
            return json.dumps(
                {
                    "status": "error",
                    "error_type": "EmptyFileError",
                    "message": f"CSV file '{path}' is empty.",
                }
            )

        raw_headers = rows[0]
        data_rows = rows[1:]
        initial_row_count = len(data_rows)

        # Normalize column headers if requested
        if params.normalize_column_names:
            headers = [_snake_case(h) for h in raw_headers]
        else:
            headers = [h.strip() for h in raw_headers]

        # Track missing values per column
        missing_per_col: dict[str, int] = {h: 0 for h in headers}

        cleaned_data: list[list[str]] = []
        empty_rows_dropped = 0

        for row in data_rows:
            # Pad row if short
            padded = row + [""] * (len(headers) - len(row))
            stripped_cells = [c.strip() for c in padded[: len(headers)]]

            all_empty = all(c == "" for c in stripped_cells)
            any_empty = any(c == "" for c in stripped_cells)

            # Update missing counters
            for i, val in enumerate(stripped_cells):
                if val == "":
                    missing_per_col[headers[i]] += 1

            if params.empty_row_strategy == "drop_all_empty" and all_empty:
                empty_rows_dropped += 1
                continue
            if params.empty_row_strategy == "drop_any_empty" and any_empty:
                empty_rows_dropped += 1
                continue

            if params.empty_row_strategy == "fill_defaults":
                filled_cells = []
                for i, c in enumerate(stripped_cells):
                    if c == "":
                        col_h = headers[i]
                        custom_fill = params.fill_values.get(col_h) if params.fill_values else None
                        filled_cells.append(custom_fill if custom_fill is not None else "N/A")
                    else:
                        filled_cells.append(c)
                stripped_cells = filled_cells

            cleaned_data.append(stripped_cells)

        # Handle duplicates
        duplicates_removed = 0
        if params.drop_duplicates:
            seen: set[tuple[str, ...]] = set()
            deduped: list[list[str]] = []

            # Determine key indices
            if params.duplicate_subset:
                norm_subset = [
                    _snake_case(col) if params.normalize_column_names else col
                    for col in params.duplicate_subset
                ]
                indices = [headers.index(c) for c in norm_subset if c in headers]
            else:
                indices = list(range(len(headers)))

            for row in cleaned_data:
                key = tuple(row[i] for i in indices)
                if key in seen:
                    duplicates_removed += 1
                else:
                    seen.add(key)
                    deduped.append(row)
            cleaned_data = deduped

        # Infer column types and gather statistics
        col_types: dict[str, str] = {}
        col_values: dict[str, list[str]] = {h: [] for h in headers}
        for row in cleaned_data:
            for i, val in enumerate(row):
                col_values[headers[i]].append(val)

        stats: dict[str, Any] = {}
        numeric_data: dict[str, list[float]] = {}

        for h in headers:
            vals = col_values[h]
            ctype = _infer_type(vals)
            col_types[h] = ctype
            if ctype in ("integer", "float"):
                nums = []
                for v in vals:
                    with contextlib.suppress(ValueError):
                        nums.append(float(v))
                if nums:
                    numeric_data[h] = nums
                    stats[h] = _calculate_numeric_stats(nums)

        # Outlier detection
        outliers_result: dict[str, Any] | None = None
        if params.detect_outliers_column:
            target_col = (
                _snake_case(params.detect_outliers_column)
                if params.normalize_column_names
                else params.detect_outliers_column
            )
            if target_col in numeric_data:
                outliers_result = {
                    "column": target_col,
                    **_detect_iqr_outliers(numeric_data[target_col]),
                }
            else:
                outliers_result = {
                    "column": target_col,
                    "error": f"Column '{target_col}' not found or is not numeric.",
                }

        # Graphical visualization
        chart: str | None = None
        vis_col = params.visualize_column
        if not vis_col and headers:
            # Pick first numerical column or first column
            numeric_cols = [h for h in headers if col_types.get(h) in ("integer", "float")]
            vis_col = numeric_cols[0] if numeric_cols else headers[0]
        else:
            vis_col = _snake_case(vis_col) if params.normalize_column_names and vis_col else vis_col

        if vis_col and vis_col in col_values:
            ctype = col_types.get(vis_col, "text")
            if ctype in ("integer", "float") and vis_col in numeric_data:
                chart = _render_numerical_histogram(vis_col, numeric_data[vis_col])
            else:
                chart = _render_categorical_chart(vis_col, col_values[vis_col])

        # Save cleaned file if requested or if modifications occurred
        saved_path: str | None = None
        if params.output_file_path:
            out_p = Path(params.output_file_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(cleaned_data)
            saved_path = str(out_p)
        else:
            # Default auto-save in workspace/cleaned_<filename>.csv
            auto_path = self._workspace_dir / f"cleaned_{path.name}"
            try:
                auto_path.parent.mkdir(parents=True, exist_ok=True)
                with open(auto_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    writer.writerows(cleaned_data)
                saved_path = str(auto_path)
            except Exception:
                saved_path = None

        # Preview first 5 rows as dictionaries
        preview = [{headers[i]: row[i] for i in range(len(headers))} for row in cleaned_data[:5]]

        result = CleanCSVResult(
            status="success",
            summary=CleanCSVSummary(
                initial_rows=initial_row_count,
                cleaned_rows=len(cleaned_data),
                columns=headers,
                duplicates_removed=duplicates_removed,
                empty_rows_removed=empty_rows_dropped,
                missing_values_per_column=missing_per_col,
            ),
            column_types=col_types,
            statistics=stats,
            outliers=outliers_result,
            visualization=chart,
            saved_to=saved_path,
            preview=preview,
        )

        return result.model_dump_json(indent=2)
