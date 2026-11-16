"""Research RAG — agentic, guardrailed, multimodal RAG over an arXiv paper corpus.

The project is run from a checkout rather than installed as a wheel, so this
makes `config/` (which lives at the repository root, beside `src/`) importable
no matter how the process was started — `uvicorn research_rag.api.app:app`,
`pytest`, or a script in `scripts/`. Without it, only the entry points that
remember to patch sys.path would work.
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if (_PROJECT_ROOT / "config").is_dir() and str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

__all__ = []
