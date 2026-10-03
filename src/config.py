"""Configuration module for Semantic Caching for LLM.

Centralizes all application configurations, embedding parameters,
similarity thresholds, and storage paths with environment variable support.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass
class AppConfig:
    """Central configuration class."""

    # Project metadata
    project_name: str = "Semantic Caching for LLM - Cost-Benefit Cascade Flow"
    version: str = "0.1.0"
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
    )

    # Embedding and Similarity Settings (Dhano's module extension point)
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    )
    similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
    )

    # Cascade and Cost-Benefit Decision Settings (Praj's module extension point)
    cascade_cost_threshold: float = field(
        default_factory=lambda: float(os.getenv("CASCADE_COST_THRESHOLD", "0.05"))
    )
    cascade_quality_threshold: float = field(
        default_factory=lambda: float(os.getenv("CASCADE_QUALITY_THRESHOLD", "0.75"))
    )

    # Storage Settings
    cache_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("CACHE_DIR", str(Path(__file__).resolve().parent.parent / "cache"))
        )
    )

    # LLM Settings (Mock/API)
    default_llm_model: str = field(
        default_factory=lambda: os.getenv("DEFAULT_LLM_MODEL", "mock-llm-v1")
    )

    def __post_init__(self) -> None:
        """Ensure cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)


# Default singleton instance for easy import across modules
config = AppConfig()
