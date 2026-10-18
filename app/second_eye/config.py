"""Configuration from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_SIDE_PX = 1536
JPEG_QUALITY = 90


def _flag(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_demo_images(value: str | None) -> list[tuple[str, str]]:
    """Parse "Label=path;Label=path" into [(label, path), ...], skipping malformed entries."""
    items: list[tuple[str, str]] = []
    for chunk in (value or "").split(";"):
        label, sep, path = chunk.partition("=")
        if sep and label.strip() and path.strip():
            items.append((label.strip(), path.strip()))
    return items


@dataclass(frozen=True)
class Settings:
    vllm_base_url: str = ""
    vllm_api_key: str = ""
    vllm_model: str = ""
    vllm_timeout: float = 90.0
    clm_url: str = ""
    clm_api_key: str = ""
    clm_timeout: float = 30.0
    clm_model: str = "harvinderbhullar/second-eye"  # display only; the server picks its own head
    demo_images: list[tuple[str, str]] = field(default_factory=list)
    mock_vllm: bool = False
    mock_clm: bool = False  # CLM_URL points at a stand-in (e.g. tests/fake_clm.py); only labels the UI
    # Display-only labels for the info chips under the X-ray viewer.
    encoder_label: str = ""
    head_label: str = ""
    served_on: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        return cls(
            vllm_base_url=env.get("VLLM_BASE_URL", "").strip().rstrip("/"),
            vllm_api_key=env.get("VLLM_API_KEY", "").strip(),
            vllm_model=env.get("VLLM_MODEL", "").strip(),
            vllm_timeout=float(env.get("VLLM_TIMEOUT", "") or 90),
            clm_url=env.get("CLM_URL", "").strip().rstrip("/"),
            clm_api_key=env.get("CLM_API_KEY", "").strip(),
            clm_timeout=float(env.get("CLM_TIMEOUT", "") or 30),
            clm_model=env.get("CLM_MODEL", "").strip() or "harvinderbhullar/second-eye",
            demo_images=_parse_demo_images(env.get("DEMO_IMAGES")),
            mock_vllm=_flag(env.get("MOCK_VLLM")),
            mock_clm=_flag(env.get("MOCK_CLM")),
            encoder_label=env.get("ENCODER_LABEL", "").strip(),
            head_label=env.get("HEAD_LABEL", "").strip(),
            served_on=env.get("SERVED_ON", "").strip(),
        )

    @property
    def clm_enabled(self) -> bool:
        return bool(self.clm_url)

    @property
    def vlm_label(self) -> str:
        """Short reader name for the UI, e.g. "Nemotron"."""
        if "nemotron" in self.vllm_model.lower():
            return "Nemotron"
        if not self.vllm_model:
            return "mock" if self.mock_vllm else "VLM"
        return self.vllm_model.rsplit("/", 1)[-1]

    @property
    def model_name(self) -> str:
        return self.vllm_model or ("mock-vlm" if self.mock_vllm else "")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Real environment variables (e.g. Space secrets) win over .env.
    load_dotenv(override=False)
    return Settings.from_env()
