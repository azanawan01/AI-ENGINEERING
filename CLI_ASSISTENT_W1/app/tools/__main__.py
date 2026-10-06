"""CLI entrypoint for inspecting assistant tools directly from terminal.

Usage:
    python -m app.tools         # Display summary of all registered tools
    python -m app.tools --demo  # Run demonstrations of all tools
"""

from __future__ import annotations

import json
import sys

from app.tools import create_default_registry, format_tools_summary


def run_demo() -> None:
    print("=" * 74)
    print("🚀 RUNNING TOOLS DEMONSTRATION")
    print("=" * 74)

    registry = create_default_registry()

    # 1. clean_csv demo
    print("\n[1/4] Testing tool: clean_csv")
    csv_input = json.dumps(
        {
            "file_path": "workspace/sample_data.csv",
            "drop_duplicates": True,
            "visualize_column": "salary",
        }
    )
    print(f"Args: {csv_input}")
    res_csv = json.loads(registry.execute("clean_csv", csv_input))
    print(f"Status: {res_csv.get('status')}")
    if "visualization" in res_csv and res_csv["visualization"]:
        print("\nGenerated ASCII Visualization:")
        print(res_csv["visualization"])

    # 2. fix_grammar demo
    print("\n" + "-" * 74)
    print("[2/4] Testing tool: fix_grammar")
    grammar_input = json.dumps(
        {
            "text": "she do not goes to school yesterday because it raining",
        }
    )
    print(f"Args: {grammar_input}")
    res_grammar = json.loads(registry.execute("fix_grammar", grammar_input))
    print(f"Status: {res_grammar.get('status')}")
    print(f"Original Text:  {res_grammar.get('original_text')}")
    print(f"Corrected Text: {res_grammar.get('corrected_text')}")
    if "corrections" in res_grammar:
        print("Corrections applied:")
        for c in res_grammar["corrections"]:
            print(f"  • [{c.get('type')}] '{c.get('original')}' -> '{c.get('replacement')}' ({c.get('explanation')})")

    # 3. translate_text demo
    print("\n" + "-" * 74)
    print("[3/4] Testing tool: translate_text")
    translate_input = json.dumps(
        {
            "text": "Good morning and welcome to our company.",
            "target_language": "French",
            "tone": "formal",
        }
    )
    print(f"Args: {translate_input}")
    res_trans = json.loads(registry.execute("translate_text", translate_input))
    print(f"Status: {res_trans.get('status')}")
    print(f"Result: {res_trans.get('translated_text')}")

    # 4. natural_language_database demo
    print("\n" + "-" * 74)
    print("[4/4] Testing tool: natural_language_database")
    db_input = json.dumps(
        {
            "database_path": "workspace/company.db",
            "query_or_question": "SELECT id, name, role, salary FROM employees LIMIT 3;",
        }
    )
    print(f"Args: {db_input}")
    res_db = json.loads(registry.execute("natural_language_database", db_input))
    print(f"Status: {res_db.get('status')}")
    if "ascii_table" in res_db and res_db["ascii_table"]:
        print("Rendered SQL Table:")
        print(res_db["ascii_table"])

    print("\n" + "=" * 74)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 74)


def main() -> int:
    registry = create_default_registry()
    if len(sys.argv) > 1 and sys.argv[1] in ("--demo", "-d"):
        run_demo()
    else:
        print(format_tools_summary(registry))
        print("\nTip: Run 'python -m app.tools --demo' to test all tools live in your terminal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
