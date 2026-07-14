"""DeepSeek LLM API Client for FDIR Autonomous Design Agents.

Reads credentials from .env file or environment variables:
  DEEPSEEK_API_KEY
  DEEPSEEK_BASE_URL (defaults to https://api.deepseek.com)

Uses standard library HTTP request (urllib.request) to eliminate heavy dependencies.
"""

from __future__ import annotations
import os
import json
import urllib.request
import urllib.error
from typing import Dict, List, Any, Optional


def load_env_file(env_path: Optional[str] = None) -> None:
    """Helper to parse a local .env file into os.environ if not already set."""
    if env_path is None:
        # Search relative to repository root
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        env_path = os.path.join(base_dir, ".env")

    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k not in os.environ:
                        os.environ[k] = v


class DeepSeekClient:
    """OpenAI-compatible client for DeepSeek LLM APIs."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: str = "deepseek-chat"):
        load_env_file()
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        self.model = model

        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not found in environment or .env file! "
                "Please ensure .env contains DEEPSEEK_API_KEY=your_key"
            )

    def chat_completion(self, messages: List[Dict[str, str]],
                        temperature: float = 0.2,
                        max_tokens: int = 1024) -> str:
        """Call DeepSeek /chat/completions endpoint."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                return resp_data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"DeepSeek API Error (HTTP {e.code}): {err_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to connect to DeepSeek API endpoint '{url}': {e}") from e
