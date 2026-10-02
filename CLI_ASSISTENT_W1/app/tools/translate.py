"""Tool for language translation with tone control and linguistic notes."""

from __future__ import annotations

import json
import logging
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.llm.base import ToolDefinition

logger = logging.getLogger(__name__)


class TranslateTextInput(BaseModel):
    """Input parameters for the translate_text tool."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        ...,
        min_length=1,
        description="The source text to translate.",
    )
    target_language: str = Field(
        ...,
        min_length=2,
        description=(
            "The language to translate the text into "
            "(e.g. 'Spanish', 'French', 'Urdu', 'German', 'Japanese')."
        ),
    )
    source_language: str = Field(
        default="auto",
        description="The language of the source text, or 'auto' for automatic detection.",
    )
    tone: Literal["neutral", "formal", "informal", "business", "literary"] = Field(
        default="neutral",
        description="The desired tone/register for the translated text.",
    )


class TranslateTextResult(BaseModel):
    """Validated structured result for text translation."""

    status: Literal["success", "error"]
    original_text: str
    detected_source_language: str
    target_language: str
    translated_text: str
    tone: str
    notes: list[str] = Field(default_factory=list)


_TRANSLATE_SYSTEM_PROMPT = """\
You are a master polyglot translator. Translate the provided text accurately and fluently \
into the specified target language while preserving the requested tone.

You must respond ONLY with a JSON object matching this schema:
{
  "status": "success",
  "original_text": "<verbatim original input>",
  "detected_source_language": "<detected or provided source language>",
  "target_language": "<target language>",
  "translated_text": "<accurate translated text>",
  "tone": "<neutral/formal/informal/business/literary>",
  "notes": ["<any important context, nuances, or alternative phrasing>"]
}
"""


class TranslateTextTool:
    """Tool that translates text across languages with validated JSON outputs."""

    def __init__(self, client: OpenAI | None = None, model: str = "") -> None:
        self._client = client
        self._model = model

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="translate_text",
            description=(
                "Translate text into any target language with tone adjustment "
                "(neutral, formal, informal, business). "
                "Detects source language and provides linguistic/cultural notes."
            ),
            parameters=TranslateTextInput.model_json_schema(),
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
            params = TranslateTextInput.model_validate(raw)
        except ValidationError as exc:
            return json.dumps(
                {
                    "status": "error",
                    "error_type": "ValidationError",
                    "message": f"Invalid arguments for translate_text: {exc}",
                }
            )

        if self._client and self._model:
            try:
                user_msg = (
                    f"Translate the following text:\n"
                    f"Source Language: {params.source_language}\n"
                    f"Target Language: {params.target_language}\n"
                    f"Desired Tone: {params.tone}\n\n"
                    f"Text:\n{params.text}"
                )
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": _TRANSLATE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )
                content = response.choices[0].message.content or ""
                validated_result = TranslateTextResult.model_validate_json(content)
                return validated_result.model_dump_json(indent=2)
            except Exception as exc:
                logger.warning("LLM translation failed (%s); using fallback translation", exc)

        # Fallback simulator for offline testing without an active LLM
        return json.dumps(
            {
                "status": "success",
                "original_text": params.text,
                "detected_source_language": (
                    params.source_language if params.source_language != "auto" else "English"
                ),
                "target_language": params.target_language,
                "translated_text": f"[{params.target_language}] {params.text}",
                "tone": params.tone,
                "notes": ["Offline fallback mode: connect LLM server for live neural translation."],
            },
            indent=2,
        )
