"""Tool for inspecting and correcting grammar, punctuation, sentence structure, and tenses."""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.llm.base import ToolDefinition

logger = logging.getLogger(__name__)


class GrammarCorrection(BaseModel):
    """Details of a single linguistic correction."""

    type: Literal["punctuation", "sentence_structure", "tense", "spelling", "word_choice"] = Field(
        ..., description="The category of the correction."
    )
    original: str = Field(..., description="The original word or phrase that contained the error.")
    replacement: str = Field(..., description="The corrected word or phrase.")
    explanation: str = Field(..., description="Clear explanation of why this change was made.")


class FixGrammarInput(BaseModel):
    """Input parameters for the fix_grammar tool."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        ...,
        min_length=1,
        description=(
            "The text to analyze and correct for grammar, punctuation, "
            "sentence structure, and tenses."
        ),
    )
    focus_areas: list[
        Literal["punctuation", "sentence_structure", "tenses", "spelling", "word_choice"]
    ] = Field(
        default=["punctuation", "sentence_structure", "tenses"],
        description="Specific aspects to focus on.",
    )
    style: Literal["standard", "formal", "casual", "academic", "concise"] = Field(
        default="standard",
        description="Desired writing style or register.",
    )


class FixGrammarResult(BaseModel):
    """Validated structured result for grammar corrections."""

    status: Literal["success", "error"]
    original_text: str
    corrected_text: str
    corrections: list[GrammarCorrection]
    grammar_score: int = Field(ge=0, le=100)
    readability_level: str
    summary: str


_GRAMMAR_SYSTEM_PROMPT = """\
You are an expert editor and grammarian. Analyze the provided user text and fix all errors \
specifically targeting: punctuation, sentence structure, tenses, and spelling.

You must respond ONLY with a JSON object matching this schema:
{
  "status": "success",
  "original_text": "<verbatim original input>",
  "corrected_text": "<fully corrected, polished version>",
  "corrections": [
    {
      "type": "punctuation" | "sentence_structure" | "tense" | "spelling" | "word_choice",
      "original": "<problematic fragment>",
      "replacement": "<corrected fragment>",
      "explanation": "<short linguistic reason>"
    }
  ],
  "grammar_score": <int 0-100 evaluating initial text quality>,
  "readability_level": "Easy" | "Intermediate" | "Advanced",
  "summary": "<brief high-level summary of revisions>"
}
"""


def _heuristic_grammar_fix(text: str) -> FixGrammarResult:
    """Fallback rule-based corrector when no LLM client is available."""
    corrections: list[GrammarCorrection] = []
    working = text.strip()

    # Rule 1: Capitalization at start of text/sentences
    if working and working[0].islower():
        orig_char = working[0]
        working = orig_char.upper() + working[1:]
        corrections.append(
            GrammarCorrection(
                type="punctuation",
                original=orig_char,
                replacement=orig_char.upper(),
                explanation="Capitalized the first letter of the sentence.",
            )
        )

    # Rule 2: Missing end punctuation
    if working and working[-1] not in ".!?":
        working += "."
        corrections.append(
            GrammarCorrection(
                type="punctuation",
                original=text[-1] if text else "",
                replacement=f"{text[-1] if text else ''}.",
                explanation="Added terminal period to complete the sentence.",
            )
        )

    # Rule 3: Spacing before punctuation (e.g. 'hello , world' -> 'hello, world')
    space_punct = re.search(r"\s+([,.;?!])", working)
    if space_punct:
        matched = space_punct.group(0)
        replacement = space_punct.group(1)
        working = re.sub(r"\s+([,.;?!])", r"\1", working)
        corrections.append(
            GrammarCorrection(
                type="punctuation",
                original=matched,
                replacement=replacement,
                explanation="Removed improper whitespace before punctuation.",
            )
        )

    # Rule 4: Common tense/agreement errors (e.g. 'do not goes', 'does not went')
    tense_patterns = [
        (
            r"\b(do|did)\s+not\s+goes\b",
            "did not go",
            "tense",
            "Auxiliary verb requires base form 'go'.",
        ),
        (
            r"\bdo\s+not\s+went\b",
            "did not go",
            "tense",
            "Negated past requires 'did not' + base form.",
        ),
        (
            r"\bis\s+goes\b",
            "goes",
            "sentence_structure",
            "Redundant auxiliary 'is' removed.",
        ),
        (
            r"\bit\s+raining\b",
            "it was raining",
            "tense",
            "Added past auxiliary 'was' for past continuous.",
        ),
    ]
    for pattern, repl, ctype, expl in tense_patterns:
        match = re.search(pattern, working, flags=re.IGNORECASE)
        if match:
            orig = match.group(0)
            working = re.sub(pattern, repl, working, flags=re.IGNORECASE)
            corrections.append(
                GrammarCorrection(
                    type=ctype,  # type: ignore[arg-type]
                    original=orig,
                    replacement=repl,
                    explanation=expl,
                )
            )

    score = max(50, 100 - (len(corrections) * 15))
    return FixGrammarResult(
        status="success",
        original_text=text,
        corrected_text=working,
        corrections=corrections,
        grammar_score=score,
        readability_level="Intermediate",
        summary=(
            f"Applied {len(corrections)} rule-based correction(s)."
            if corrections
            else "Text appears grammatically sound."
        ),
    )


class FixGrammarTool:
    """Tool that inspects and repairs grammar, punctuation, sentence structure, and tenses."""

    def __init__(self, client: OpenAI | None = None, model: str = "") -> None:
        self._client = client
        self._model = model

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="fix_grammar",
            description=(
                "Analyze and fix grammar, punctuation, sentence structure, and tenses "
                "in the provided text. Returns a validated breakdown of every correction "
                "with explanations and overall score."
            ),
            parameters=FixGrammarInput.model_json_schema(),
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
            params = FixGrammarInput.model_validate(raw)
        except ValidationError as exc:
            return json.dumps(
                {
                    "status": "error",
                    "error_type": "ValidationError",
                    "message": f"Invalid arguments for fix_grammar: {exc}",
                }
            )

        # If an OpenAI client is provided, query the LLM with structured JSON output
        if self._client and self._model:
            try:
                user_msg = (
                    f"Please analyze and fix the grammar in this text.\n"
                    f"Focus areas: {', '.join(params.focus_areas)}\n"
                    f"Style: {params.style}\n\n"
                    f"Text:\n{params.text}"
                )
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": _GRAMMAR_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                content = response.choices[0].message.content or ""
                # Validate the LLM output against our Pydantic model
                validated_result = FixGrammarResult.model_validate_json(content)
                return validated_result.model_dump_json(indent=2)
            except Exception as exc:
                logger.warning(
                    "LLM grammar check failed (%s); falling back to heuristic engine", exc
                )

        # Fallback to rule-based engine
        fallback = _heuristic_grammar_fix(params.text)
        return fallback.model_dump_json(indent=2)
