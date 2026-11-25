"""Loader for the self-contained Web Control Console (/_console) static HTML asset."""
from __future__ import annotations

from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=1)
def render_console_html() -> bytes:
    return resources.files("basalt_colab.web").joinpath("console.html").read_bytes()
