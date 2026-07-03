"""Unified LLM router — switch between Ollama, Anthropic, OpenAI, and any
other provider supported by LiteLLM. Public API unchanged from the
hand-rolled version, so call sites in agent.py / rag.py / Stock Tracker.py
keep working with zero changes.

The rest of the app calls `llm_chat(messages, ...)` and never has to know
which backend is in use. The active backend is selected via:

  - Sidebar dropdown -> `set_backend("ollama:llama3.2")` (preferred path)
  - Env var ``LLM_BACKEND`` as a fallback for non-Streamlit contexts
  - Falls back to ``DEFAULT_BACKEND`` (Ollama Llama 3.2) if nothing is set

Cloud backends auto-disable when their respective API keys are missing so
the local Ollama path keeps working with zero config.

Public API
----------
- ``BACKENDS`` — ordered registry: id -> {label, provider, model, ...}
- ``available_backends()`` — list of ids whose provider is currently usable
- ``set_backend(id)`` / ``get_backend()`` — module-level current selection
- ``llm_chat(messages, *, stream=False, format=None, backend=None)``
    Returns the full response text (stream=False) or an iterator of text
    chunks (stream=True). Always returns *strings*, never provider objects,
    so call sites stay provider-agnostic.

Implementation note
-------------------
Provider dispatch goes through LiteLLM, which gives us a single function
(``litellm.completion``) for every provider on the planet. The previous
hand-rolled Anthropic client is gone — LiteLLM handles streaming, system
prompts, and JSON mode internally. To add a new provider (Gemini, Groq,
Mistral, Together…), append one row to ``BACKENDS``. No code changes.
"""
from __future__ import annotations

import json
import os
import re
from typing import Iterator, Optional

# LiteLLM network-call timeout (seconds). Network stalls would otherwise
# hang the whole UI — agent.py uses a 60 s per-step timeout, so 120 s here
# gives the LLM call comfortable headroom while still bailing out on dead
# connections.
_LLM_TIMEOUT_S = 120.0

# Best-effort .env load. Safe to import even if python-dotenv is missing.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------
# Each backend's ``model`` string follows LiteLLM's "provider/model" format —
# that's all LiteLLM needs to dispatch the call. Provider keys ("anthropic",
# "openai", etc.) are also used by ``available_backends()`` to gate options
# behind the presence of the right API-key env var.
#
# Ordered so the sidebar shows local-free first, then cheap-cloud, then premium.
BACKENDS: dict[str, dict] = {
    "ollama:llama3.2": {
        "label": "Ollama · Llama 3.2 (local · free)",
        "provider": "ollama",
        "model": "ollama/llama3.2",
        "supports_json": True,
    },
    "anthropic:haiku-3-5": {
        "label": "Claude Haiku 3.5 (fast · cheap)",
        "provider": "anthropic",
        "model": "anthropic/claude-3-5-haiku-20241022",
        "supports_json": False,  # we instruct via prompt instead
    },
    "anthropic:sonnet-3-5": {
        "label": "Claude Sonnet 3.5 (high quality)",
        "provider": "anthropic",
        "model": "anthropic/claude-3-5-sonnet-20241022",
        "supports_json": False,
    },
    "anthropic:sonnet-4-5": {
        "label": "Claude Sonnet 4.5 (frontier reasoning)",
        "provider": "anthropic",
        "model": "anthropic/claude-sonnet-4-5",
        "supports_json": False,
    },
    "anthropic:opus-4-5": {
        "label": "Claude Opus 4.5 (deepest synthesis · slow)",
        "provider": "anthropic",
        "model": "anthropic/claude-opus-4-5",
        "supports_json": False,
    },
    # OpenAI / Gemini etc. are one line each — uncomment when keys arrive:
    # "openai:gpt-4o":          {"label": "GPT-4o",          "provider": "openai", "model": "gpt-4o",            "supports_json": True},
    # "gemini:1-5-pro":         {"label": "Gemini 1.5 Pro",  "provider": "gemini", "model": "gemini/gemini-1.5-pro", "supports_json": True},
}

DEFAULT_BACKEND = "ollama:llama3.2"

# max_tokens for cloud completions. 8192 covers a 10-section analytical memo
# at 4 000–7 000 tokens with comfortable headroom.
_DEFAULT_MAX_TOKENS = 8192

# Provider -> env-var name. Used by ``available_backends()`` so the sidebar
# never offers a backend that will fail the moment it's selected.
_PROVIDER_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "gemini":    "GEMINI_API_KEY",
    "groq":      "GROQ_API_KEY",
    "mistral":   "MISTRAL_API_KEY",
    "together":  "TOGETHER_API_KEY",
    # ollama is local — no key required.
}


# ---------------------------------------------------------------------------
# Module-level selection
# ---------------------------------------------------------------------------

_current_backend: str = os.environ.get("LLM_BACKEND") or DEFAULT_BACKEND


def set_backend(backend_id: str) -> None:
    """Set the active backend for subsequent ``llm_chat`` calls.

    Silently ignores unknown ids so a stale session_state value can't break
    the app — falls back to whatever was already set.
    """
    global _current_backend
    if backend_id in BACKENDS:
        _current_backend = backend_id


def get_backend() -> str:
    return _current_backend


def has_provider_key(provider: str) -> bool:
    """True iff the env var for ``provider`` is set (or the provider is local)."""
    env_var = _PROVIDER_KEY_ENV.get(provider)
    if env_var is None:
        return True  # local providers (ollama) need no key
    return bool((os.environ.get(env_var) or "").strip())


def has_anthropic_key() -> bool:
    """Back-compat shim — Stock Tracker.py still calls this directly."""
    return has_provider_key("anthropic")


def available_backends() -> list[str]:
    """Return backend ids whose provider is currently usable.

    Cloud options are filtered out when their API key is missing so the
    sidebar dropdown doesn't show options that will fail.
    """
    return [bid for bid, meta in BACKENDS.items()
            if has_provider_key(meta["provider"])]


# ---------------------------------------------------------------------------
# LiteLLM dispatch
# ---------------------------------------------------------------------------
# A single ``litellm.completion`` call replaces every per-provider client we
# used to maintain. LiteLLM converts the OpenAI-style payload into whatever
# wire format the destination provider expects, so we only have to think
# about *one* request shape.

_LITELLM_INITIALIZED = False


def _ensure_litellm():
    """Import LiteLLM lazily and apply a few sane defaults the first time.

    ``drop_params=True`` makes LiteLLM silently drop OpenAI-style params that
    the destination provider doesn't support, instead of erroring out — this
    matters because some of our agent calls send ``response_format``-style
    arguments and we'd rather they no-op on Ollama than crash.
    """
    global _LITELLM_INITIALIZED
    import litellm  # local import keeps cold-start cheap
    if not _LITELLM_INITIALIZED:
        litellm.drop_params = True
        # Quiet LiteLLM's default verbose console logging.
        litellm.suppress_debug_info = True
        _LITELLM_INITIALIZED = True
    return litellm


def _format_messages_for_json(messages: list[dict]) -> list[dict]:
    """Inject a system-level JSON nudge for providers without first-class JSON."""
    nudge = (
        "Respond with ONLY a single valid JSON object. No prose, no "
        "markdown fences, no commentary outside the JSON."
    )
    out: list[dict] = []
    seen_system = False
    for m in messages:
        if m.get("role") == "system":
            out.append({
                "role": "system",
                "content": (m.get("content", "") + "\n\n" + nudge).strip(),
            })
            seen_system = True
        else:
            out.append(m)
    if not seen_system:
        out.insert(0, {"role": "system", "content": nudge})
    return out


def _strip_code_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public unified API
# ---------------------------------------------------------------------------

def llm_chat(
    messages: list[dict],
    *,
    stream: bool = False,
    format: Optional[str] = None,
    backend: Optional[str] = None,
):
    """Send ``messages`` (OpenAI-style: list of {role, content}) to the active
    backend and return the text response.

    Parameters
    ----------
    messages : list[dict]
        Standard chat messages. ``role`` may be "system", "user", or
        "assistant".
    stream : bool
        If True, returns an iterator yielding *text chunks* (strings).
        If False, returns the full response text as a single string.
    format : Optional[str]
        Set to ``"json"`` to request JSON output. LiteLLM passes this through
        as ``response_format`` for providers that support it (OpenAI, Ollama)
        and we additionally inject a system-prompt nudge for providers that
        don't (Anthropic).
    backend : Optional[str]
        Override the active backend for this single call. Falls back to
        ``get_backend()`` (set via the sidebar / ``set_backend``).

    Returns
    -------
    str | Iterator[str]
    """
    backend_id = backend or _current_backend
    meta = BACKENDS.get(backend_id) or BACKENDS[DEFAULT_BACKEND]
    provider = meta["provider"]
    model = meta["model"]

    # Hard-fail-soft: if the chosen cloud backend has no key, fall back to
    # local Ollama rather than raising deep inside a tab.
    if not has_provider_key(provider):
        return llm_chat(
            messages,
            stream=stream,
            format=format,
            backend=DEFAULT_BACKEND,
        )

    litellm = _ensure_litellm()

    # Prepare messages — for JSON requests on providers without native support
    # we add a system-level instruction so the model still produces clean JSON.
    msgs = list(messages)
    if format == "json" and not meta.get("supports_json", False):
        msgs = _format_messages_for_json(msgs)

    kwargs: dict = {
        "model": model,
        "messages": msgs,
        "stream": stream,
        "max_tokens": _DEFAULT_MAX_TOKENS,
        "timeout": _LLM_TIMEOUT_S,
    }
    if format == "json" and meta.get("supports_json", False):
        kwargs["response_format"] = {"type": "json_object"}

    resp = litellm.completion(**kwargs)

    if stream:
        def _gen():
            for chunk in resp:
                # LiteLLM normalises every provider into OpenAI-shaped chunks:
                # chunk.choices[0].delta.content
                try:
                    delta = chunk.choices[0].delta
                    txt = getattr(delta, "content", None) or ""
                except Exception:
                    txt = ""
                if txt:
                    yield txt
        return _gen()

    # Non-streaming: pull the full message content out.
    try:
        text = resp.choices[0].message.content or ""
    except Exception:
        text = ""
    text = text.strip()
    if format == "json":
        text = _strip_code_fences(text)
    return text


def llm_json(messages: list[dict], *, backend: Optional[str] = None) -> dict | None:
    """Convenience wrapper that calls ``llm_chat`` with ``format="json"``
    and parses the result. Returns ``None`` on parse failure.
    """
    text = llm_chat(messages, format="json", backend=backend)
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
