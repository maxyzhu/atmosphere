"""
Configuration and secrets management.

This module centralizes access to environment variables and secrets.
It exists so that:
    - Tokens are loaded once, not scattered across the codebase
    - Missing tokens fail loudly with a helpful message (not silently)
    - Tests can mock the token without touching the real .env

Convention: sensitive values are loaded lazily on first call, not at
import time. This lets tests override them without needing to exist
before atmosphere is imported.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv


# Load .env once when this module is imported. If .env doesn't exist,
# load_dotenv is silent — which is what we want for CI / environments
# where env vars come from elsewhere.
load_dotenv()


class ConfigError(RuntimeError):
    """Raised when a required configuration value is missing or malformed."""


@lru_cache(maxsize=1)
def get_mapillary_token() -> str:
    """
    Return the Mapillary API access token, loaded from MAPILLARY_ACCESS_TOKEN.

    Raises:
        ConfigError: If the token is missing or obviously invalid.

    The result is cached for the process lifetime; tests that need to
    override it should clear the cache or patch the environment variable
    before the first call.
    """
    token = os.getenv("MAPILLARY_ACCESS_TOKEN", "").strip()

    if not token or token == "your_token_here":
        raise ConfigError(
            "MAPILLARY_ACCESS_TOKEN is not set. "
            "Copy .env.example to .env and fill in your token from "
            "https://www.mapillary.com/dashboard/developers"
        )

    # Minimal sanity check: Mapillary v4 client tokens begin with "MLY|".
    # This is not a security check — just a typo guard.
    if not token.startswith("MLY|"):
        raise ConfigError(
            f"MAPILLARY_ACCESS_TOKEN does not look like a valid v4 client token "
            f"(expected prefix 'MLY|'). Got: {token[:6]}..."
        )

    return token