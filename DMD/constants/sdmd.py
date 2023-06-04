from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class SVDTypes(Enum):
    PARTIAL: str = "partial"
    TRUNCATED: str = "truncated"


@dataclass(frozen=True)
class ModsTypes(Enum):
    STANDARD: str = "standard"
    EXACT: str = "exact"
    EXACT_SCALED: str = "exact_scaled"