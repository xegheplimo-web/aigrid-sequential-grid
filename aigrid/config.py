"""Service configuration, including the exchange snapshot location."""

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from .settings import AIGridSettings as BaseSettings
from .settings import load_settings as load_base_settings


@dataclass(frozen=True)
class AIGridSettings(BaseSettings):
    exchange_info_path: str = ""


def load_settings(path: str | Path | None = None) -> AIGridSettings:
    base = load_base_settings(path)
    snapshot = ""
    if path is not None and Path(path).exists():
        with Path(path).open("rb") as stream:
            snapshot = tomllib.load(stream).get("aigrid", {}).get("exchange_info_path", "")
        if not isinstance(snapshot, str):
            raise ValueError("exchange_info_path must be a string")
    return AIGridSettings(
        **{item.name: getattr(base, item.name) for item in fields(base)},
        exchange_info_path=snapshot,
    )
