"""API sub-routers package.

Currently exposes the main `oratory` router. Additional domain routers can be
added here and re-exported for inclusion in the FastAPI `app`.
"""

from .oratory import router  # noqa: F401

__all__ = [
    "router",
]
