"""Tool for querying SQLite databases: sql generation -> validation -> execution -> result."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.llm.base import ToolDefinition

logger = logging.getLogger(__name__)


class DatabaseQueryInput(BaseModel):
    """Input parameters for natural_language_database tool."""

    model_config = ConfigDict(extra="forbid")

    query_or_question: str = Field(
        ...,
        min_length=1,
        description=(
            "A natural language question (e.g. 'Show top 5 highest paid employees') "
            "OR a direct SQL query (e.g. 'SELECT name, salary FROM employees')."
        ),
    )
    database_path: str = Field(
        default="workspace/company.db",
        description="Path to the SQLite database file.",
    )
    action: Literal["query", "describe_schema"] = Field(
        default="query",
        description=(
            "Action to perform: 'query' (generates, validates and executes SQL) "
            "or 'describe_schema' (lists all tables and column types)."
        ),
    )
    max_rows: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of rows to return.",
    )


class DatabaseQueryResult(BaseModel):
    """Validated output from natural_language_database."""

    status: Literal["success", "validation_error", "execution_error"]
    database_path: str
    generated_sql: str
    validation_passed: bool
    validation_message: str
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    execution_time_ms: float
    ascii_table: str


_BLOCKED_SQL_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|TRUNCATE|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)


def _render_ascii_table(columns: list[str], rows: list[list[Any]]) -> str:
    """Format tabular data into a neat ASCII box table."""
    if not columns:
        return "Empty result set."

    # Compute column widths
    str_rows = [[str(cell if cell is not None else "NULL") for cell in row] for row in rows]
    widths = [len(col) for col in columns]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # Cap column width at 40 for terminal readability
    widths = [min(w, 40) for w in widths]

    sep_line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    header_line = (
        "| " + " | ".join(col.ljust(widths[i])[: widths[i]] for i, col in enumerate(columns)) + " |"
    )

    lines = [sep_line, header_line, sep_line]
    for row in str_rows:
        row_str = (
            "| "
            + " | ".join(cell.ljust(widths[i])[: widths[i]] for i, cell in enumerate(row))
            + " |"
        )
        lines.append(row_str)
    lines.append(sep_line)
    return "\n".join(lines)


def _introspect_schema(conn: sqlite3.Connection) -> dict[str, list[dict[str, str]]]:
    """Retrieve database tables and their column definitions."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    )
    tables = [row[0] for row in cursor.fetchall()]
    schema: dict[str, list[dict[str, str]]] = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info('{table}');")
        cols = cursor.fetchall()
        schema[table] = [
            {"name": col[1], "type": col[2], "notnull": str(bool(col[3])), "pk": str(bool(col[5]))}
            for col in cols
        ]
    return schema


def _format_schema_for_prompt(schema: dict[str, list[dict[str, str]]]) -> str:
    lines = []
    for table, cols in schema.items():
        col_desc = ", ".join(f"{c['name']} {c['type']}" for c in cols)
        lines.append(f"Table '{table}': ({col_desc})")
    return "\n".join(lines)


class NaturalLanguageDatabaseTool:
    """Tool that queries SQLite databases through the 4-stage pipeline:

    SQL generation -> Validation -> Execution -> Result.
    """

    def __init__(
        self,
        client: OpenAI | None = None,
        model: str = "",
        workspace_dir: str = "workspace",
    ) -> None:
        self._client = client
        self._model = model
        self._workspace_dir = Path(workspace_dir)

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="natural_language_database",
            description=(
                "Query SQLite databases using natural language or SQL. "
                "Executes the safe 4-stage pipeline: "
                "SQL generation -> validation -> execution -> tabular result. "
                "Can also inspect schemas of tables."
            ),
            parameters=DatabaseQueryInput.model_json_schema(),
        )

    def execute(self, arguments_json: str) -> str:
        try:
            raw = json.loads(arguments_json)
        except Exception as exc:
            return json.dumps(
                {
                    "status": "validation_error",
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
            params = DatabaseQueryInput.model_validate(raw)
        except ValidationError as exc:
            return json.dumps(
                {
                    "status": "validation_error",
                    "error_type": "ValidationError",
                    "message": f"Invalid arguments for natural_language_database: {exc}",
                }
            )

        db_path = Path(params.database_path)
        if not db_path.is_file():
            candidate = self._workspace_dir / db_path.name
            if candidate.is_file():
                db_path = candidate
            else:
                return json.dumps(
                    {
                        "status": "execution_error",
                        "error_type": "FileNotFoundError",
                        "message": f"Database file not found at '{params.database_path}'.",
                    }
                )

        try:
            conn = sqlite3.connect(str(db_path))
        except Exception as exc:
            return json.dumps(
                {
                    "status": "execution_error",
                    "message": f"Failed to connect to SQLite database: {exc}",
                }
            )

        try:
            schema = _introspect_schema(conn)

            # Stage 0: Schema Inspection Action
            if params.action == "describe_schema":
                schema_str = _format_schema_for_prompt(schema)
                return json.dumps(
                    {
                        "status": "success",
                        "database_path": str(db_path),
                        "action": "describe_schema",
                        "schema": schema,
                        "schema_text": schema_str,
                    },
                    indent=2,
                )

            # Stage 1: SQL Generation
            input_text = params.query_or_question.strip()
            # If the user passed raw SQL, evaluate it directly rather than generating
            sql_keywords = (
                r"^\s*(SELECT|PRAGMA|EXPLAIN|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|"
                r"DETACH|TRUNCATE)\b"
            )
            is_direct_sql = bool(re.match(sql_keywords, input_text, re.IGNORECASE))

            if is_direct_sql:
                candidate_sql = input_text.rstrip(";")
            else:
                candidate_sql = self._generate_sql(input_text, schema)

            # Stage 2: Validation
            validation_passed, validation_msg = self._validate_sql(candidate_sql, conn)
            if not validation_passed:
                result = DatabaseQueryResult(
                    status="validation_error",
                    database_path=str(db_path),
                    generated_sql=candidate_sql,
                    validation_passed=False,
                    validation_message=validation_msg,
                    columns=[],
                    rows=[],
                    row_count=0,
                    execution_time_ms=0.0,
                    ascii_table=f"Validation failed: {validation_msg}",
                )
                return result.model_dump_json(indent=2)

            # Stage 3: Execution
            start_t = time.perf_counter()
            cursor = conn.cursor()
            if not re.search(r"\bLIMIT\b", candidate_sql, re.IGNORECASE):
                exec_sql = f"{candidate_sql} LIMIT {params.max_rows}"
            else:
                exec_sql = candidate_sql
            cursor.execute(exec_sql)
            raw_rows = cursor.fetchall()
            exec_time = (time.perf_counter() - start_t) * 1000.0

            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            dict_rows = [{columns[i]: row[i] for i in range(len(columns))} for row in raw_rows]

            # Stage 4: Result Formatter
            ascii_table = _render_ascii_table(columns, raw_rows)

            result = DatabaseQueryResult(
                status="success",
                database_path=str(db_path),
                generated_sql=candidate_sql,
                validation_passed=True,
                validation_message="Query passed syntax, schema, and read-only validation.",
                columns=columns,
                rows=dict_rows,
                row_count=len(dict_rows),
                execution_time_ms=round(exec_time, 2),
                ascii_table=ascii_table,
            )
            return result.model_dump_json(indent=2)
        finally:
            conn.close()

    def _generate_sql(self, question: str, schema: dict[str, list[dict[str, str]]]) -> str:
        """Translate natural language to SQLite query using schema."""
        schema_desc = _format_schema_for_prompt(schema)

        if self._client and self._model:
            prompt = (
                f"You are a SQLite database assistant. Given the following database schema:\n"
                f"{schema_desc}\n\n"
                f"Generate a single read-only SQLite SELECT query that answers this question:\n"
                f"'{question}'\n\n"
                f"Return ONLY the SQL query text. Do not wrap in markdown or backticks."
            )
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Return only a single safe SQLite SELECT query without formatting."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )
                sql = response.choices[0].message.content or ""
                sql = re.sub(r"^```(sql)?", "", sql.strip(), flags=re.IGNORECASE)
                sql = re.sub(r"```$", "", sql.strip())
                return sql.strip().rstrip(";")
            except Exception as exc:
                logger.warning("LLM SQL generation failed: %s; using heuristic", exc)

        # Fallback heuristic: basic SELECT from first available table
        first_table = next(iter(schema.keys())) if schema else "data"
        return f"SELECT * FROM {first_table}"

    def _validate_sql(self, sql: str, conn: sqlite3.Connection) -> tuple[bool, str]:
        """Validate that SQL is safe, read-only, and syntactically valid."""
        # Check 1: Must not be empty
        if not sql.strip():
            return False, "Query cannot be empty."

        # Check 2: Disallow multiple statements (semicolon injection)
        cleaned = sql.strip().rstrip(";")
        if ";" in cleaned:
            return False, "Multiple SQL statements separated by semicolons are strictly forbidden."

        # Check 3: Check for mutating keywords
        if _BLOCKED_SQL_KEYWORDS.search(cleaned):
            match = _BLOCKED_SQL_KEYWORDS.search(cleaned)
            blocked_word = match.group(0) if match else "MUTATION"
            return (
                False,
                f"Mutating query rejected: keyword '{blocked_word}' is forbidden. "
                "Only read-only queries are permitted.",
            )

        # Check 4: Must start with SELECT, PRAGMA, or EXPLAIN
        if not re.match(r"^\s*(SELECT|PRAGMA|EXPLAIN)\b", cleaned, re.IGNORECASE):
            return False, "Query must begin with a read-only keyword (SELECT, PRAGMA, or EXPLAIN)."

        # Check 5: Syntax & schema validation using SQLite EXPLAIN
        try:
            cursor = conn.cursor()
            cursor.execute(f"EXPLAIN {cleaned}")
        except sqlite3.OperationalError as exc:
            return False, f"SQLite validation error: {exc}"
        except Exception as exc:
            return False, f"Unexpected SQL validation error: {exc}"

        return True, "Query successfully validated."
