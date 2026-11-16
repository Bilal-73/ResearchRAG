# config/

Single source of truth for all environment-driven configuration.

- **`settings.py`** — a `pydantic-settings` `Settings` class that reads `.env` and
  exposes typed config (Azure OpenAI, separate VLM, Qdrant, retrieval knobs).

No other module reads environment variables directly — everything takes its config via
constructor injection from a `Settings` instance. This keeps every class testable with
plain Python values instead of real credentials.

## Usage
```python
from config.settings import get_settings
settings = get_settings()
```
