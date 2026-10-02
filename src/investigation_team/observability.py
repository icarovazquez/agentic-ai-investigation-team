"""
Langfuse client setup.

Same Colab-independence rule as llm.py: this module reads plain
environment variables. The notebook is responsible for pulling the
actual key values from Colab's userdata and assigning them to
os.environ before importing investigation_team.
"""

from __future__ import annotations

import os

from langfuse import Langfuse


def _require_env(name: str) -> str:
    value = os.environ.get(name)

    if not value:
        raise RuntimeError(
            f"Environment variable '{name}' is not set. "
            "If you're running this in Colab, pull it from "
            "userdata and assign it to os.environ before "
            "importing investigation_team."
        )

    return value


LANGFUSE_PUBLIC_KEY = _require_env("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = _require_env("LANGFUSE_SECRET_KEY")
LANGFUSE_BASE_URL = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    base_url=LANGFUSE_BASE_URL,
)


def check_connection() -> None:
    """
    Verify the Langfuse connection and print the result.

    Not called automatically on import — call this explicitly from
    the notebook if you want a visible confirmation, the way the
    original notebook's setup cell did.
    """

    try:
        projects = langfuse.api.projects.get()
        print("✅ Connected. Projects:", [p.name for p in projects.data])
    except Exception as e:
        print("❌ Langfuse connection failed")
        print("Type:", type(e).__name__)
        print("Message:", str(e))
