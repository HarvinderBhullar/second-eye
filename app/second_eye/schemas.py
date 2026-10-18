"""Data shapes: the model's reading, the JSON schema sent to vLLM, and API responses."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

FractureSuspected = Literal["yes", "no", "uncertain"]
Confidence = Literal["low", "medium", "high"]
BodyRegion = Literal["hand_wrist", "leg_knee_ankle_foot", "hip_pelvis", "shoulder", "other", "unclear"]
ImageQuality = Literal["good", "limited", "poor"]
Agreement = Literal["agree", "disagree", "uncertain"]


class Reading(BaseModel):
    model_config = ConfigDict(extra="ignore")

    fracture_suspected: FractureSuspected
    confidence: Confidence  # categorical on purpose: no fake probabilities
    body_region: BodyRegion
    hardware_visible: bool
    image_quality: ImageQuality
    findings: str
    reasoning: str
    limitations: str


# Written out by hand (rather than Reading.model_json_schema()) so it stays flat and
# portable across guided-decoding backends.
READING_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "fracture_suspected": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "body_region": {
            "type": "string",
            "enum": ["hand_wrist", "leg_knee_ankle_foot", "hip_pelvis", "shoulder", "other", "unclear"],
        },
        "hardware_visible": {"type": "boolean", "description": "Plates, screws or nails visible."},
        "image_quality": {"type": "string", "enum": ["good", "limited", "poor"]},
        "findings": {"type": "string", "description": "Max 2 sentences, plain language, what is seen and where."},
        "reasoning": {"type": "string", "description": "Max 3 sentences."},
        "limitations": {"type": "string", "description": "Max 2 sentences, e.g. single view, low resolution."},
    },
    "required": [
        "fracture_suspected",
        "confidence",
        "body_region",
        "hardware_visible",
        "image_quality",
        "findings",
        "reasoning",
        "limitations",
    ],
    "additionalProperties": False,
}

RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {"name": "fracture_reading", "schema": READING_JSON_SCHEMA, "strict": True},
}


class ClmResult(BaseModel):
    p: float | None = None
    threshold: float | None = None
    flagged: bool | None = None
    latency_ms: float | None = None
    error: str | None = None


class AnalyzeResponse(BaseModel):
    reading: Reading | None
    parse_error: bool = False
    raw: str
    model: str
    latency_ms: float
    total_ms: float
    mock: bool
    clm: ClmResult | None = None
    agreement: Agreement | None = None


class ServiceHealth(BaseModel):
    ok: bool
    model_listed: bool | None = None
    models: list[str] = []
    error: str | None = None
    status_code: int | None = None


class HealthResponse(BaseModel):
    vllm: ServiceHealth
    clm: ServiceHealth | None
    mock: bool
