"""Makes `src/` importable for the CLI scripts and initialises logging once.

The real composition root is `research_rag.bootstrap` — it lives inside the
package so the API server can import it too. This module only exists to let
`python scripts/foo.py` work from a checkout, and re-exports the builders so
the scripts have a single import line.
"""
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for path in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Answers, citations and paper titles routinely contain non-ASCII (the "›" in a
# citation, en-dashes, Greek letters in equations). A Windows console defaults to
# cp1252 and raises UnicodeEncodeError on all of it, which would crash a script
# *after* it had done the expensive work. Force UTF-8 and never fail on output.
for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):   # a redirected or already-closed stream
            pass

def _require_project_interpreter() -> None:
    """Fail with an explanation, not a 20-frame ImportError.

    On Windows `python` usually resolves to a system install that has none of this
    project's dependencies, so the real problem is the interpreter, not the package."""
    import importlib.util

    needed = {"pydantic_settings": "pydantic-settings", "fastapi": "fastapi",
              "qdrant_client": "qdrant-client", "langgraph": "langgraph"}
    missing = [pip for module, pip in needed.items() if importlib.util.find_spec(module) is None]
    if not missing:
        return
    venv_python = _PROJECT_ROOT / (".venv/Scripts/python.exe" if sys.platform == "win32"
                                   else ".venv/bin/python")
    print(f"\n  Missing package(s): {', '.join(missing)}", file=sys.stderr)
    print(f"  Running under:      {sys.executable}", file=sys.stderr)
    if ".venv" not in sys.executable and venv_python.exists():
        print("\n  That is not this project's virtualenv. Run it with:", file=sys.stderr)
        print(f'      "{venv_python}" {" ".join(sys.argv)}', file=sys.stderr)
    else:
        print("\n  Install them with:  pip install -r requirements.txt", file=sys.stderr)
    print("", file=sys.stderr)
    sys.exit(1)


_require_project_interpreter()

from config.logging_config import setup_logging
from config.settings import get_settings

setup_logging(get_settings().log_level)

from research_rag.bootstrap import (  # noqa: E402 — must follow the sys.path setup above
    build_agent, build_index_components, build_ingestion, build_query_components, build_rag_system,
)

logger = logging.getLogger(__name__)

__all__ = ["build_agent", "build_index_components", "build_ingestion", "build_query_components",
           "build_rag_system",
           "get_settings", "PROJECT_ROOT"]

PROJECT_ROOT = _PROJECT_ROOT
