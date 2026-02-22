# Copyright 2026. Model tier mapping for providers.

PROVIDER_MODELS: dict[str, dict[str, str]] = {
    "claude": {"reasoning": "opus", "balanced": "sonnet", "fast": "haiku"},
    "codex": {"reasoning": "o3", "balanced": "gpt-4.1", "fast": "gpt-4.1-mini"},
}
