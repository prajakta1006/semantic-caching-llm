"""Central configuration for the semantic caching project."""

from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass
class AppConfig:
    """Application-wide configuration."""

    # ---------------------------------------------------------
    # Project
    # ---------------------------------------------------------

    project_name: str = (
        "Semantic Caching for LLM - Cost and Latency Optimization"
    )

    version: str = "0.2.0"

    debug: bool = field(
        default_factory=lambda: os.getenv(
            "DEBUG",
            "false",
        ).lower() in ("true", "1", "yes")
    )

    # ---------------------------------------------------------
    # Semantic cache configuration
    # Owned by TanTan / used through CacheInterface
    # ---------------------------------------------------------

    embedding_model_name: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL_NAME",
            "all-MiniLM-L6-v2",
        )
    )

    similarity_threshold: float = field(
        default_factory=lambda: float(
            os.getenv(
                "SIMILARITY_THRESHOLD",
                "0.85",
            )
        )
    )

    # ---------------------------------------------------------
    # Cache storage
    # ---------------------------------------------------------

    cache_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "CACHE_DIR",
                str(
                    Path(__file__).resolve().parent.parent
                    / "cache"
                ),
            )
        )
    )

    # ---------------------------------------------------------
    # LLM configuration
    # Owned by Dhano
    # ---------------------------------------------------------

    llm_provider: str = field(
        default_factory=lambda: os.getenv(
            "LLM_PROVIDER",
            "mock",
        )
    )

    default_llm_model: str = field(
        default_factory=lambda: os.getenv(
            "DEFAULT_LLM_MODEL",
            "mock-llm-v1",
        )
    )

    llm_api_key_env: str = field(
        default_factory=lambda: os.getenv(
            "LLM_API_KEY_ENV",
            "LLM_API_KEY",
        )
    )

    llm_timeout_seconds: float = field(
        default_factory=lambda: float(
            os.getenv(
                "LLM_TIMEOUT_SECONDS",
                "30",
            )
        )
    )

    def __post_init__(self) -> None:
        """Validate configuration and prepare storage."""

        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                "SIMILARITY_THRESHOLD must be between 0.0 and 1.0"
            )

        if self.llm_timeout_seconds <= 0:
            raise ValueError(
                "LLM_TIMEOUT_SECONDS must be greater than 0"
            )

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


config = AppConfig()
