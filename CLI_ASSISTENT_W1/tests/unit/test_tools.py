import json
import sqlite3
from pathlib import Path

import pytest

from app.tools.base import ToolRegistry
from app.tools.clean_csv import CleanCSVTool
from app.tools.database import NaturalLanguageDatabaseTool
from app.tools.grammar import FixGrammarTool
from app.tools.translate import TranslateTextTool


@pytest.fixture
def temp_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "test_data.csv"
    csv_file.write_text(
        "id, name, department, salary, age\n"
        "1, Alice, Engineering, 80000, 30\n"
        "2, Bob, Marketing, 60000, 35\n"
        "3, Charlie, Engineering, 85000, 28\n"
        "4, David, Sales, 62000, 32\n"
        "5, Emma, Marketing, 65000, 29\n"
        "6, Frank, Sales, 61000, 31\n"
        "1, Alice, Engineering, 80000, 30\n"  # Duplicate
        ",,,,\n"  # Empty row
        "7, George, HR, 58000, \n"  # Missing age
        "8, Outlier, Engineering, 900000, 40\n",
        encoding="utf-8",
    )
    return csv_file


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "test.db"
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, role TEXT, score REAL);")
    cur.executemany(
        "INSERT INTO users VALUES (?, ?, ?, ?)",
        [
            (1, "Alice", "Admin", 95.5),
            (2, "Bob", "User", 82.0),
            (3, "Charlie", "User", 88.5),
        ],
    )
    conn.commit()
    conn.close()
    return db_file


def test_clean_csv_removes_duplicates_and_empty_rows(temp_csv: Path) -> None:
    tool = CleanCSVTool(workspace_dir=str(temp_csv.parent))
    args = json.dumps(
        {
            "file_path": str(temp_csv),
            "drop_duplicates": True,
            "empty_row_strategy": "drop_any_empty",
        }
    )
    res = json.loads(tool.execute(args))

    assert res["status"] == "success"
    summary = res["summary"]
    assert summary["initial_rows"] == 10
    assert summary["duplicates_removed"] == 1
    assert summary["empty_rows_removed"] == 2  # The all-empty row + the row with missing age
    assert summary["cleaned_rows"] == 7


def test_clean_csv_generates_visualizations_and_detects_outliers(temp_csv: Path) -> None:
    tool = CleanCSVTool(workspace_dir=str(temp_csv.parent))
    args = json.dumps(
        {
            "file_path": str(temp_csv),
            "visualize_column": "department",
            "detect_outliers_column": "salary",
            "empty_row_strategy": "drop_all_empty",
        }
    )
    res = json.loads(tool.execute(args))

    assert res["status"] == "success"
    assert "Category Distribution: department" in res["visualization"]
    assert res["outliers"] is not None
    assert res["outliers"]["outlier_count"] == 1
    assert 900000.0 in res["outliers"]["outlier_values"]


def test_clean_csv_handles_missing_file() -> None:
    tool = CleanCSVTool()
    res = json.loads(tool.execute(json.dumps({"file_path": "nonexistent_file.csv"})))
    assert res["status"] == "error"
    assert res["error_type"] == "FileNotFoundError"


def test_fix_grammar_heuristics() -> None:
    tool = FixGrammarTool()
    args = json.dumps(
        {
            "text": "she do not goes to school yesterday because it raining",
        }
    )
    res = json.loads(tool.execute(args))

    assert res["status"] == "success"
    assert res["original_text"] == "she do not goes to school yesterday because it raining"
    assert "did not go" in res["corrected_text"]
    assert len(res["corrections"]) >= 2
    assert res["grammar_score"] <= 100


def test_translate_text_fallback() -> None:
    tool = TranslateTextTool()
    args = json.dumps(
        {
            "text": "Good morning",
            "target_language": "French",
        }
    )
    res = json.loads(tool.execute(args))

    assert res["status"] == "success"
    assert res["target_language"] == "French"
    assert "Good morning" in res["original_text"]


def test_database_describe_schema(temp_db: Path) -> None:
    tool = NaturalLanguageDatabaseTool(workspace_dir=str(temp_db.parent))
    args = json.dumps(
        {
            "database_path": str(temp_db),
            "action": "describe_schema",
            "query_or_question": "describe",
        }
    )
    res = json.loads(tool.execute(args))

    assert res["status"] == "success"
    assert "users" in res["schema"]
    col_names = [col["name"] for col in res["schema"]["users"]]
    assert "name" in col_names
    assert "score" in col_names


def test_database_executes_valid_query_and_formats_table(temp_db: Path) -> None:
    tool = NaturalLanguageDatabaseTool(workspace_dir=str(temp_db.parent))
    args = json.dumps(
        {
            "database_path": str(temp_db),
            "query_or_question": "SELECT name, role, score FROM users ORDER BY score DESC",
        }
    )
    res = json.loads(tool.execute(args))

    assert res["status"] == "success"
    assert res["validation_passed"] is True
    assert res["row_count"] == 3
    assert res["rows"][0]["name"] == "Alice"
    assert "+-------+-------+-------+" in res["ascii_table"] or "|" in res["ascii_table"]


def test_database_rejects_mutations(temp_db: Path) -> None:
    tool = NaturalLanguageDatabaseTool(workspace_dir=str(temp_db.parent))

    # Reject DROP TABLE
    res_drop = json.loads(
        tool.execute(
            json.dumps(
                {
                    "database_path": str(temp_db),
                    "query_or_question": "DROP TABLE users;",
                }
            )
        )
    )
    assert res_drop["status"] == "validation_error"
    assert res_drop["validation_passed"] is False
    assert "forbidden" in res_drop["validation_message"].lower()

    # Reject DELETE FROM
    res_del = json.loads(
        tool.execute(
            json.dumps(
                {
                    "database_path": str(temp_db),
                    "query_or_question": "DELETE FROM users WHERE id = 1;",
                }
            )
        )
    )
    assert res_del["status"] == "validation_error"
    assert res_del["validation_passed"] is False


def test_database_rejects_semicolon_chaining(temp_db: Path) -> None:
    tool = NaturalLanguageDatabaseTool(workspace_dir=str(temp_db.parent))
    res = json.loads(
        tool.execute(
            json.dumps(
                {
                    "database_path": str(temp_db),
                    "query_or_question": "SELECT * FROM users; SELECT * FROM users",
                }
            )
        )
    )
    assert res["status"] == "validation_error"
    assert "Multiple SQL statements" in res["validation_message"]


def test_tool_registry_dispatch() -> None:
    registry = ToolRegistry([FixGrammarTool()])
    assert len(registry.definitions) == 1
    assert registry.definitions[0].name == "fix_grammar"

    # Execution of existing tool
    res = json.loads(registry.execute("fix_grammar", json.dumps({"text": "hello world"})))
    assert res["status"] == "success"

    # Execution of nonexistent tool
    res_missing = json.loads(registry.execute("unknown_tool", "{}"))
    assert res_missing["status"] == "error"
    assert res_missing["error_type"] == "ToolNotFound"
