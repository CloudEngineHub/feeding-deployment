from typing import Any, Dict, List, Optional
from openai import (
    APIConnectionError as OpenAIAPIConnectionError,
    APIStatusError as OpenAIAPIStatusError,
    RateLimitError as OpenAIRateLimitError,
)
from anthropic import (
    APIConnectionError as AnthropicAPIConnectionError,
    APIStatusError as AnthropicAPIStatusError,
    RateLimitError as AnthropicRateLimitError,
)
import time
import os
import feeding_deployment.preference_learning.config as root_config  # type: ignore
from feeding_deployment.preference_learning.config.preference_bundle import (
    PREFERENCE_BUNDLE as _PREF_DIMS,
    COLOR_FIELDS,
    NAV_OFFSET_FIELDS,
    TEXT_FIELDS,
    format_color,
    format_nav_offset,
)
PREF_FIELDS: List[str] = [name for (name, _, _) in root_config.PREFERENCE_BUNDLE]
_COLOR_FIELD_SET = set(COLOR_FIELDS)
_NAV_OFFSET_FIELD_SET = set(NAV_OFFSET_FIELDS)
_TEXT_FIELD_SET = set(TEXT_FIELDS)


def _format_pref_value(field: str, value: Any) -> str:
    """Stringify a bundle value for episode text / comparisons. Color dims and
    nav-offset dims get the stable compact encoding regardless of whether the
    value arrives as a canonical dict or an already-formatted "h=..,s=.."/"dx=.."
    string (the eval loop pins truth values as formatted strings); everything
    else is str()."""
    if field in _COLOR_FIELD_SET:
        return format_color(value)
    if field in _NAV_OFFSET_FIELD_SET:
        return format_nav_offset(value)
    return str(value)


def _pred_matches_truth(field: str, pred_val: Any, truth_val: Any) -> bool:
    """Field-aware equality between a predicted value (dict or string) and a
    truth value (formatted string from _extract_truth_bundle, or dict). Both
    sides are canonicalized through _format_pref_value so color/nav dims never
    compare a dict against its string encoding. Text dims tolerate
    case/whitespace differences only."""
    a = _format_pref_value(field, pred_val)
    b = _format_pref_value(field, truth_val)
    if field in _TEXT_FIELD_SET:
        return a.strip().lower() == b.strip().lower()
    return a == b

def _retry_on_rate_limit(fn, max_retries: int = 5, base_wait: float = 60.0):
    """Retry transient API failures with exponential backoff: rate limits
    (429), server-side errors (5xx, incl. 529 Overloaded), and connection
    drops. Non-transient 4xx status errors (auth, bad request) raise
    immediately -- retrying those can never succeed."""
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return fn()
        except (OpenAIRateLimitError, AnthropicRateLimitError) as e:
            last_err = e
            label = "Rate limit"
            wait = base_wait * (2 ** attempt)
        except (AnthropicAPIStatusError, OpenAIAPIStatusError) as e:
            if e.status_code < 500:
                raise
            last_err = e
            label = f"Server error {e.status_code}"
            # 5xx/overloaded usually clears much faster than a rate-limit
            # window; start small so a blip doesn't stall the run for minutes.
            wait = min(base_wait, 10.0 * (2 ** attempt))
        except (AnthropicAPIConnectionError, OpenAIAPIConnectionError) as e:
            last_err = e
            label = "Connection error"
            wait = min(base_wait, 10.0 * (2 ** attempt))
        if attempt == max_retries - 1:
            raise last_err
        print(f"  [{label}] Waiting {wait:.0f}s before retry ({attempt + 1}/{max_retries}) ...", flush=True)
        time.sleep(wait)
    raise last_err  # type: ignore[misc]


def _resolve_api_key(cli_key: Optional[str]) -> str:
    if cli_key and cli_key.strip():
        return cli_key.strip()
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    try:
        import generate_dataset_llm  # type: ignore
        file_key = getattr(generate_dataset_llm, "OPENAI_API_KEY", None)
        if isinstance(file_key, str) and file_key.strip():
            return file_key.strip()
    except Exception:
        pass
    raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY env var or pass --api-key.")


def _resolve_anthropic_key(cli_key: Optional[str] = None) -> str:
    if cli_key and cli_key.strip():
        return cli_key.strip()
    env_key = os.getenv("ANTHROPIC_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    raise RuntimeError("Anthropic API key not found. Set ANTHROPIC_API_KEY env var or pass --api-key.")


def _episode_text(day: int, context: Dict[str, Any], prefs: Dict[str, str]) -> str:
    ctx = (
        f"day={day}; meal={context.get('meal')}; setting={context.get('setting')}; "
        f"time_of_day={context.get('time_of_day')};"
    )
    # Only render fields present in the bundle: deployment always passes the
    # full bundle, but offline eval may pass a subset (--exclude-missing-dims)
    # and backfilling defaults here would plant fabricated values in memory.
    pref_str = "; ".join(f"{k}={_format_pref_value(k, prefs[k])}" for k in PREF_FIELDS if k in prefs)
    return f"{ctx}\npreferences: {pref_str}"


def _extract_truth_bundle(day_rec: Dict[str, Any]) -> Dict[str, str]:
    prefs = day_rec.get("preferences", {}) or {}
    out: Dict[str, str] = {}
    for field in PREF_FIELDS:
        val = prefs.get(field, {})
        choice = val.get("choice", "") if isinstance(val, dict) else ""
        if field in _COLOR_FIELD_SET or field in _NAV_OFFSET_FIELD_SET:
            # Canonical compact encoding (dict-valued in the dataset); str()
            # would yield a Python dict repr that nothing else can match.
            out[field] = _format_pref_value(field, choice)
        else:
            out[field] = str(choice).strip()
    return out
