# -*- coding: utf-8 -*-
"""
Robust OpenAI wrapper with:
- raw response diagnostics when JSON parsing fails upstream
- response_format/json_schema graceful fallback
- retry with exponential backoff on transient errors/timeouts
- safer message extraction (content/tool_calls/function_call)
"""

import os
import time
import json
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import openai
from openai import APIError, APIStatusError, APITimeoutError, APIConnectionError

# ---- Env ----
load_dotenv()
_DEFAULT_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
_DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
_DEFAULT_TIMEOUT = int(os.getenv("OPENAI_HTTP_TIMEOUT_SEC", "300"))
_DEFAULT_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
_DEFAULT_BACKOFF = float(os.getenv("OPENAI_RETRY_BACKOFF", "1.6"))


def _has_v1_suffix(url: Optional[str]) -> bool:
    if not url:
        return False
    return url.rstrip("/").endswith("/v1")


def _maybe_warn_base_url(url: Optional[str]):
    if not url:
        print("⚠️ OPENAI_BASE_URL not set; using OpenAI default endpoint.")
        return
    if not _has_v1_suffix(url):
        print(f"⚠️ OPENAI_BASE_URL='{url}' does not appear to end with '/v1'. "
              "If this is a model aggregator/gateway, make sure the path is correct.")


def _strip_code_fence(s: str) -> str:
    """Remove a single markdown code fence wrapper if present."""
    if not s:
        return s
    t = s.strip()
    if t.startswith("```"):
        t = t.strip("`")
        nl = t.find("\n")
        t = t[nl + 1:] if nl != -1 else t
    return t.strip()


class OpenAIWrapper:
    """
    Minimal but robust wrapper around client.chat.completions.create
    - .chat(...) returns a dict with:
        - model, label, messages
        - content (string)
        - tool_calls (list of dicts {function:{name, arguments}})
    """
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        label: Optional[str] = None,
        allow_response_format: bool = False,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        backoff: float = _DEFAULT_BACKOFF,
        debug: bool = False,
    ):
        self.model = model
        self.label = label or model
        self.allow_response_format = allow_response_format
        self.api_key = api_key or _DEFAULT_API_KEY
        self.base_url = base_url or _DEFAULT_BASE_URL
        self.timeout = timeout or _DEFAULT_TIMEOUT
        self.max_retries = max_retries
        self.backoff = backoff
        self.debug = debug
        
        if not self.api_key:
            raise RuntimeError("API key not set (set OPENROUTER_API_KEY or OPENAI_API_KEY)")

        _maybe_warn_base_url(self.base_url)

        # Create client
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    # ----------- Internal helpers -----------

    def _create_once(self, kwargs: Dict[str, Any]):
        """
        Use with_raw_response for best diagnostics, then parse.
        """
        raw = self.client.chat.completions.with_raw_response.create(**kwargs)
        try:
            # If server returns valid JSON, this should succeed
            parsed = raw.parse()
            return parsed, raw
        except Exception as e:
            # Try to dump raw HTTP info to debug upstream response issues
            try:
                http = raw.http_response  # httpx.Response
                status = getattr(http, "status_code", None)
                url = getattr(http, "url", "")
                text = (getattr(http, "text", "") or "")[:800]
                print(f"\n[DEBUG] JSON parse failed. HTTP {status} {url}")
                print("[DEBUG] First 800 chars of body:\n", text)
            except Exception:
                pass
            raise e

    def _should_retry(self, exc: Exception) -> bool:
        """Classify transient errors for retry."""
        msg = str(exc)
        # OpenAI SDK exceptions
        if isinstance(exc, (APITimeoutError, APIConnectionError)):
            return True
        if isinstance(exc, APIStatusError):
            try:
                status = exc.status_code
                if status in (408, 409, 429, 500, 502, 503, 504):
                    return True
            except Exception:
                return True  # be permissive
        if isinstance(exc, APIError):
            return True

        # Generic signal from aggregator: sometimes returns HTML/empty body → JSONDecodeError inside SDK
        # We detect keywords
        low = msg.lower()
        if any(k in low for k in ["jsondecodeerror", "expecting value", "parse", "invalid json"]):
            return True

        return False

    def _maybe_strip_schema_and_retry(self, kwargs: Dict[str, Any], err: Exception):
        msg = str(err)
        trigger = any(s in msg for s in ["response_format", "json_schema", "Unsupported", "schema"])
        if trigger and "response_format" in kwargs:
            if self.debug:
                print(f"⚠️ [{self.model}] response_format/json_schema rejected, retry without schema. Err: {msg[:160]}...")
            kwargs2 = dict(kwargs)
            kwargs2.pop("response_format", None)
            return kwargs2
        return None

    # ----------- Public API -----------

    def chat(
        self,
        messages: List[Dict[str, str]],
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:

        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **kwargs
        }

        # Convert "functions" → "tools"
        if functions:
            tools = [{"type": "function", "function": func} for func in functions]
        if tools:
            call_kwargs["tools"] = tools

        if tool_choice:
            if isinstance(tool_choice, str) and not tool_choice.startswith("auto"):
                call_kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice}
                }
            else:
                call_kwargs["tool_choice"] = tool_choice

        # Only pass response_format if explicitly allowed
        if response_format and self.allow_response_format:
            if "schema" in response_format:
                call_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": response_format["schema"],
                        "strict": response_format.get("strict", False),
                    },
                }
            else:
                call_kwargs["response_format"] = response_format

        if temperature is not None:
            call_kwargs["temperature"] = temperature

        last_err = None
        # Retry loop
        for attempt in range(1, self.max_retries + 1):
            try:
                resp, raw = self._create_once(call_kwargs)
                # --- Normal path ---
                msg = resp.choices[0].message

                result = {
                    "model": self.model,
                    "label": self.label,
                    "messages": messages,
                    "content": (getattr(msg, "content", "") or "").strip(),
                    "tool_calls": [],
                }

                # Tool calls (tools API)
                if getattr(msg, "tool_calls", None):
                    result["tool_calls"] = [
                        {"function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ]

                # Legacy function_call (very rare on new SDKs, but keep)
                if not result["tool_calls"] and getattr(msg, "function_call", None):
                    fc = msg.function_call
                    result["tool_calls"] = [{
                        "function": {
                            "name": getattr(fc, "name", None),
                            "arguments": getattr(fc, "arguments", None)
                        }
                    }]

                # If no content but has tool_calls, try best-effort extract
                if not result["content"] and result["tool_calls"]:
                    best = max(
                        (tc for tc in result["tool_calls"]),
                        key=lambda t: len((t["function"].get("arguments") or ""))
                    )
                    args = best["function"].get("arguments")
                    if isinstance(args, str):
                        s = _strip_code_fence(args)
                        result["content"] = s
                    elif isinstance(args, (dict, list)):
                        result["content"] = json.dumps(args, ensure_ascii=False)
                    else:
                        result["content"] = (args or "")

                return result

            except Exception as e:
                # First, see if schema was the issue → drop response_format and retry once immediately
                alt_kwargs = self._maybe_strip_schema_and_retry(call_kwargs, e)
                if alt_kwargs is not None:
                    call_kwargs = alt_kwargs
                    # try schema-removed call in the SAME attempt
                    try:
                        resp, raw = self._create_once(call_kwargs)
                        msg = resp.choices[0].message
                        result = {
                            "model": self.model,
                            "label": self.label,
                            "messages": messages,
                            "content": (getattr(msg, "content", "") or "").strip(),
                            "tool_calls": [],
                        }
                        if getattr(msg, "tool_calls", None):
                            result["tool_calls"] = [
                                {"function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                for tc in msg.tool_calls
                            ]
                        if not result["tool_calls"] and getattr(msg, "function_call", None):
                            fc = msg.function_call
                            result["tool_calls"] = [{
                                "function": {
                                    "name": getattr(fc, "name", None),
                                    "arguments": getattr(fc, "arguments", None)
                                }
                            }]
                        if not result["content"] and result["tool_calls"]:
                            best = max(
                                (tc for tc in result["tool_calls"]),
                                key=lambda t: len((t["function"].get("arguments") or ""))
                            )
                            args = best["function"].get("arguments")
                            if isinstance(args, str):
                                s = _strip_code_fence(args)
                                result["content"] = s
                            elif isinstance(args, (dict, list)):
                                result["content"] = json.dumps(args, ensure_ascii=False)
                            else:
                                result["content"] = (args or "")
                        return result
                    except Exception as e2:
                        # Fall through to generic retry logic
                        e = e2

                last_err = e
                transient = self._should_retry(e)
                if self.debug:
                    print(f"[WARN] {self.model} attempt {attempt}/{self.max_retries} failed: {e}")

                if attempt >= self.max_retries or not transient:
                    # Give up
                    raise e

                # Exponential backoff
                time.sleep(self.backoff ** attempt)

        # Should not reach here
        raise last_err