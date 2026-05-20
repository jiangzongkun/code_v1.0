import json
import os
from typing import Any, Dict, List, Optional

try:
    from .openai_client import chat_completion, response_content, response_usage
except ModuleNotFoundError as exc:
    if exc.name not in {"prompting.openai_client", "openai_client"}:
        raise
    import httpx
    from openai import OpenAI

    DEFAULT_MODEL = "Qwen/Qwen3.5-27B"
    LEGACY_DEFAULT_MODELS = {"gpt-4", "gpt4", "llama3.3:latest"}
    FALSE_VALUES = {"0", "false", "False", "no", "NO"}

    _client: Optional[OpenAI] = None
    _client_config = None

    def _normalize_base_url(base_url: str) -> str:
        base_url = base_url.strip().rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        return base_url

    def resolve_base_url() -> str:
        base_url = (
            os.environ.get("OPENAI_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("PROXY_BASE")
        )
        if not base_url:
            base_url = "http://localhost:11434/v1"
        return _normalize_base_url(base_url)

    def resolve_api_key() -> str:
        return (
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("VLLM_API_KEY")
            or "ollama"
        )

    def resolve_model(llm_source: Optional[str] = None) -> str:
        env_model = (
            os.environ.get("OPENAI_MODEL")
            or os.environ.get("LLM_MODEL")
            or os.environ.get("OLLAMA_MODEL")
        )
        if env_model:
            return env_model
        if llm_source and llm_source not in LEGACY_DEFAULT_MODELS:
            return llm_source
        return DEFAULT_MODEL

    def _get_client() -> OpenAI:
        global _client, _client_config

        base_url = resolve_base_url()
        api_key = resolve_api_key()
        timeout = float(os.environ.get("OPENAI_TIMEOUT", "180"))
        config = (base_url, api_key, timeout)
        if _client is None or _client_config != config:
            _client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                http_client=httpx.Client(
                    timeout=httpx.Timeout(timeout),
                    trust_env=False,
                ),
            )
            _client_config = config
        return _client

    def _extra_body_for_model(model: str) -> Optional[Dict[str, Any]]:
        extra_body = os.environ.get("OPENAI_EXTRA_BODY")
        if extra_body:
            return json.loads(extra_body)

        disable_thinking = os.environ.get("OPENAI_DISABLE_THINKING", "1")
        enable_thinking = os.environ.get("LLM_ENABLE_THINKING")
        if enable_thinking is not None:
            disable_thinking = "0" if enable_thinking not in FALSE_VALUES else "1"
        if disable_thinking not in FALSE_VALUES and "qwen" in model.lower():
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return None

    def chat_completion(
        messages: List[Dict[str, str]],
        llm_source: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0,
    ) -> Any:
        model = resolve_model(llm_source)
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        extra_body = _extra_body_for_model(model)
        if extra_body:
            kwargs["extra_body"] = extra_body
        return _get_client().chat.completions.create(**kwargs)

    def response_content(response: Any) -> Optional[str]:
        if not response.choices:
            return None
        return response.choices[0].message.content

    def response_usage(response: Any) -> Dict[str, Any]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if isinstance(usage, dict):
            return usage
        return dict(usage)
