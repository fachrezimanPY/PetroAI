"""
AI Client dengan multi-provider support:
- Local: Ollama (offline)
- Cloud: Anthropic Claude, OpenAI-compatible (Groq, OpenRouter, Together AI, dll)
"""
import json
import urllib.request
import urllib.error
from typing import Callable, Generator
from PyQt6.QtCore import QThread, pyqtSignal
import config


# ------------------------------------------------------------------ helpers
def _http_post(url: str, headers: dict, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _http_post_stream(url: str, headers: dict, body: dict) -> Generator[str, None, None]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if line.startswith("data: "):
                chunk = line[6:]
                if chunk and chunk != "[DONE]":
                    yield chunk


# ------------------------------------------------------------------ providers
PROVIDERS = {
    "local_ollama": {
        "label": "Local — Ollama",
        "needs_key": False,
        "default_model": "llama3.1",
        "model_examples": ["llama3.1", "llama3.2", "mistral", "gemma3", "qwen2.5"],
    },
    "anthropic": {
        "label": "Cloud — Anthropic Claude",
        "needs_key": True,
        "default_model": "claude-sonnet-4-6",
        "model_examples": ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        "base_url": "https://api.anthropic.com/v1/messages",
    },
    "openai": {
        "label": "Cloud — OpenAI",
        "needs_key": True,
        "default_model": "gpt-4o",
        "model_examples": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "base_url": "https://api.openai.com/v1/chat/completions",
    },
    "groq": {
        "label": "Cloud — Groq (cepat & gratis)",
        "needs_key": True,
        "default_model": "llama-3.1-70b-versatile",
        "model_examples": ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
    },
    "openrouter": {
        "label": "Cloud — OpenRouter (akses banyak model)",
        "needs_key": True,
        "default_model": "meta-llama/llama-3.1-70b-instruct",
        "model_examples": [
            "meta-llama/llama-3.1-70b-instruct",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-pro-1.5",
            "mistralai/mistral-7b-instruct",
        ],
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
    },
    "together": {
        "label": "Cloud — Together AI",
        "needs_key": True,
        "default_model": "meta-llama/Llama-3-70b-chat-hf",
        "model_examples": ["meta-llama/Llama-3-70b-chat-hf", "mistralai/Mixtral-8x7B-Instruct-v0.1"],
        "base_url": "https://api.together.xyz/v1/chat/completions",
    },
}


# ------------------------------------------------------------------ connection check
def check_ollama_connection() -> tuple[bool, list[str]]:
    try:
        import ollama
        response = ollama.list()
        models = [m.model for m in response.models]
        return True, models
    except Exception:
        return False, []


# ------------------------------------------------------------------ stream thread
class StreamThread(QThread):
    token_received = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, messages: list[dict], provider: str, model: str, api_key: str = ""):
        super().__init__()
        self.messages = messages
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self._full = ""

    def run(self):
        try:
            if self.provider == "local_ollama":
                self._run_ollama()
            elif self.provider == "anthropic":
                self._run_anthropic()
            else:
                self._run_openai_compat()
            self.finished.emit(self._full)
        except Exception as e:
            self.error.emit(str(e))

    # -- Ollama (local)
    def _run_ollama(self):
        import ollama as oll
        stream = oll.chat(model=self.model, messages=self.messages, stream=True)
        for chunk in stream:
            token = chunk.message.content or ""
            self._full += token
            self.token_received.emit(token)

    # -- Anthropic Claude API
    def _run_anthropic(self):
        system_msgs = [m["content"] for m in self.messages if m["role"] == "system"]
        user_msgs = [m for m in self.messages if m["role"] != "system"]

        body = {
            "model": self.model,
            "max_tokens": 2048,
            "stream": True,
            "messages": user_msgs,
        }
        if system_msgs:
            body["system"] = "\n".join(system_msgs)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        url = PROVIDERS["anthropic"]["base_url"]
        for raw_chunk in _http_post_stream(url, headers, body):
            try:
                obj = json.loads(raw_chunk)
                delta = obj.get("delta", {})
                token = delta.get("text", "")
                if token:
                    self._full += token
                    self.token_received.emit(token)
            except Exception:
                pass

    # -- OpenAI-compatible (OpenAI, Groq, OpenRouter, Together)
    def _run_openai_compat(self):
        pdef = PROVIDERS[self.provider]
        url = pdef["base_url"]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://petroai.local"
            headers["X-Title"] = "PetroAI"

        body = {
            "model": self.model,
            "messages": self.messages,
            "stream": True,
            "max_tokens": 2048,
        }
        for raw_chunk in _http_post_stream(url, headers, body):
            try:
                obj = json.loads(raw_chunk)
                choices = obj.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        self._full += token
                        self.token_received.emit(token)
            except Exception:
                pass


# ------------------------------------------------------------------ main client
class AIClient:
    def __init__(self):
        self.provider = "local_ollama"
        self.model = PROVIDERS["local_ollama"]["default_model"]
        self.api_key = ""
        self._api_keys: dict[str, str] = {}
        self._active_thread: StreamThread | None = None

    def configure(self, provider: str, model: str, api_key: str = ""):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        if api_key:
            self._api_keys[provider] = api_key

    def get_stored_key(self, provider: str) -> str:
        return self._api_keys.get(provider, "")

    def chat_stream(
        self,
        messages: list[dict],
        on_token: Callable[[str], None],
        on_done: Callable[[str], None],
        on_error: Callable[[str], None],
    ) -> StreamThread:
        thread = StreamThread(messages, self.provider, self.model, self.api_key)
        thread.token_received.connect(on_token)
        thread.finished.connect(on_done)
        thread.error.connect(on_error)
        thread.start()
        self._active_thread = thread
        return thread

    def is_local(self) -> bool:
        return self.provider == "local_ollama"

    def provider_label(self) -> str:
        return PROVIDERS.get(self.provider, {}).get("label", self.provider)

    def needs_api_key(self) -> bool:
        return PROVIDERS.get(self.provider, {}).get("needs_key", False)
