import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# The suite is fully offline (Azure and Qdrant are faked), but importing the API
# app builds a Settings object, which requires the Azure fields. Placeholders let
# the tests run on a fresh clone or in CI without a .env file. A real .env, or a
# variable already set in the environment, still takes precedence.
for _name, _value in {
    "AZURE_OPENAI_EMBEDDING_ENDPOINT": "https://example.invalid",
    "AZURE_OPENAI_EMBEDDING_API_KEY": "test-key",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-3-small",
    "AZURE_OPENAI_CHAT_ENDPOINT": "https://example.invalid",
    "AZURE_OPENAI_CHAT_API_KEY": "test-key",
    "AZURE_OPENAI_CHAT_DEPLOYMENT": "gpt-4o-mini",
}.items():
    os.environ.setdefault(_name, _value)
