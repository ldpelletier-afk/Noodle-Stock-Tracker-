"""cache.py — Streamlit-agnostic caching decorator.

Drop-in replacement for @st.cache_data. Inside a live Streamlit session it
delegates to st.cache_data (cross-rerun memoisation with automatic TTL).
Outside Streamlit (CLI scripts, tests, batch jobs) it falls back to a simple
in-process TTL dict cache — identical call signature so no code changes needed
in the callers.

Usage (same as @st.cache_data)::

    from cache import cache_data

    @cache_data(ttl=300)
    def expensive_fetch(ticker: str) -> dict: ...

    # works in Streamlit app AND in plain Python scripts
"""
import functools
import time


def _in_streamlit() -> bool:
    """Return True when a Streamlit script is actively running."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


class _TTLCache:
    """Minimal TTL-aware in-process cache backing the fallback path."""

    __slots__ = ("_store", "_ttl")

    def __init__(self, ttl: float | None):
        self._store: dict = {}
        self._ttl   = ttl

    def get(self, key):
        entry = self._store.get(key)
        if entry is None:
            return _MISSING
        value, ts = entry
        if self._ttl is not None and (time.monotonic() - ts) >= self._ttl:
            del self._store[key]
            return _MISSING
        return value

    def set(self, key, value):
        self._store[key] = (value, time.monotonic())

    def clear(self):
        self._store.clear()


_MISSING = object()


def cache_data(func=None, *, ttl: float | int | None = None, show_spinner=True, **_kw):
    """Decorator factory — mirrors the st.cache_data(ttl=...) signature.

    * Inside Streamlit → delegates to st.cache_data for proper multi-user
      per-session caching and the progress spinner.
    * Outside Streamlit → uses _TTLCache (process-wide, thread-safe enough
      for sequential scripts; not safe for multi-threaded batch workers).
    """
    def decorator(f):
        _fallback_cache = _TTLCache(float(ttl) if ttl is not None else None)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if _in_streamlit():
                import streamlit as st
                cached_fn = st.cache_data(f, ttl=ttl, show_spinner=show_spinner)
                return cached_fn(*args, **kwargs)

            key = (args, tuple(sorted(kwargs.items())))
            result = _fallback_cache.get(key)
            if result is _MISSING:
                result = f(*args, **kwargs)
                _fallback_cache.set(key, result)
            return result

        def clear():
            _fallback_cache.clear()
            # Also attempt to bust the Streamlit-side cache if we're in a
            # session — handles the case where the function was cached by
            # st.cache_data in a previous rerun.
            try:
                if _in_streamlit():
                    import streamlit as st
                    st.cache_data(f, ttl=ttl, show_spinner=show_spinner).clear()
            except Exception:
                pass

        wrapper.clear = clear
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def cache_resource(func=None, **_kw):
    """Mirror of st.cache_resource — singleton per process.

    Inside Streamlit → delegates to st.cache_resource (singleton across all
    sessions in the server process).
    Outside Streamlit → uses functools.lru_cache(maxsize=1) so the resource
    is created once per process and reused.
    """
    def decorator(f):
        _lru_cached = functools.lru_cache(maxsize=None)(f)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if _in_streamlit():
                import streamlit as st
                return st.cache_resource(f)(*args, **kwargs)
            return _lru_cached(*args, **kwargs)

        wrapper.clear = _lru_cached.cache_clear
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
