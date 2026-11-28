"""Single source of truth for generating ready-to-use AI client configs (CLI + /_console)."""
from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from typing import Any

from basalt_colab.config import Config
from basalt_colab.paths import Paths

VALID_CLIENTS = ("dsh", "claude", "codex", "opencode", "openai")
CLIENT_ALIASES = {
    "dsh": "dsh",
    "harness": "dsh",
    "deepseek-harness": "dsh",
    "deepseek": "dsh",
    "claude": "claude",
    "claude-code": "claude",
    "codex": "codex",
    "opencode": "opencode",
    "openai": "openai",
    "cursor": "openai",
    "generic": "openai",
}


def resolve_client(name: str) -> str | None:
    return CLIENT_ALIASES.get(name.strip().lower())


@dataclass(frozen=True, slots=True)
class ClientEndpoint:
    host: str
    port: int
    api_key: str
    model: str
    quant: str
    effort: str
    max_context: int
    vision: str
    key_file: str = ""

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def openai_base_url(self) -> str:
        return f"{self.base_url}/v1"

    @property
    def vision_enabled(self) -> bool:
        return self.vision != "off"

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        api_key: str,
        *,
        port: int | None = None,
        live_status: dict[str, Any] | None = None,
        paths: Paths | None = None,
    ) -> ClientEndpoint:
        st = live_status or {}
        return cls(
            host=cfg.proxy.host,
            port=int(port if port is not None else cfg.proxy.port),
            api_key=api_key,
            model=str(st.get("model") or cfg.basalt.effective_model_alias),
            quant=str(st.get("quant") or cfg.basalt.quant),
            effort=str(st.get("effort") or cfg.basalt.effort),
            max_context=int(st.get("max_context") or cfg.basalt.max_context),
            vision=str(st.get("vision") or cfg.basalt.vision),
            key_file=str(paths.key_file) if paths is not None else "",
        )


def dsh_effort_default(effort: str) -> str:
    """Maps Basalt effort (low|high|max|1-100) to DeepSeek-V4.1's native levels: off, low (=50), high (=75), max (=100)."""
    e = str(effort).strip().lower()
    if e in ("low", "high", "max"):
        return e
    if e in ("none", "off"):
        return "off"
    if e.isdigit():
        n = int(e)
        if n <= 60:
            return "low"
        if n <= 85:
            return "high"
        return "max"
    return "low"


def env_vars(ep: ClientEndpoint) -> tuple[tuple[str, str], ...]:
    """Environment variables compatible with OpenAI, Anthropic, Claude Code, Codex, and DeepSeek Harness."""
    return (
        ("OPENAI_BASE_URL", ep.openai_base_url),
        ("OPENAI_API_KEY", ep.api_key),
        ("OPENAI_MODEL", ep.model),
        ("ANTHROPIC_BASE_URL", ep.base_url),
        ("ANTHROPIC_API_KEY", ep.api_key),
        ("ANTHROPIC_AUTH_TOKEN", ep.api_key),
        ("ANTHROPIC_MODEL", ep.model),
        ("ANTHROPIC_SMALL_FAST_MODEL", ep.model),
        ("BASALT_API_KEY", ep.api_key),
    )


def format_env_lines(pairs: tuple[tuple[str, str], ...], *, powershell: bool = False) -> str:
    lines = []
    for k, v in pairs:
        if powershell:
            lines.append(f"$env:{k} = '{v.replace(chr(39), chr(39) * 2)}'")
        else:
            lines.append(f"export {k}={shlex.quote(v)}")
    return "\n".join(lines)


def opencode_config_dict(ep: ClientEndpoint, *, key_mode: str = "env") -> dict[str, Any]:
    """Builds opencode.json dict. key_mode='file' references state/api_key; 'env' uses {env:OPENAI_API_KEY} (safe to share/commit)."""
    image = (
        {"attachment": True, "modalities": {"input": ["text", "image"], "output": ["text"]}}
        if ep.vision_enabled
        else {}
    )
    if key_mode == "file" and ep.key_file:
        clean_key = ep.key_file.replace("\\", "/")
        api_key_val = f"{{file:{clean_key}}}"
    else:
        api_key_val = "{env:OPENAI_API_KEY}"
    models = {
        ep.model: dict(
            {
                "name": f"DeepSeek-V4.1-Flash {ep.quant.upper()} (Basalt, effort {ep.effort})",
                "tool_call": True,
                "reasoning": True,
                "interleaved": {"field": "reasoning_content"},
                "options": {"reasoningEffort": str(ep.effort)},
                "limit": {"context": ep.max_context, "output": 32768},
            },
            **image,
        )
    }
    return {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "basalt": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "DeepSeek-V4.1-Flash on Colab (Basalt)",
                "options": {
                    "baseURL": ep.openai_base_url,
                    "apiKey": api_key_val,
                    "headerTimeout": 1800000,
                    "chunkTimeout": 900000,
                },
                "models": models,
            }
        },
        "model": f"basalt/{ep.model}",
        "small_model": f"basalt/{ep.model}",
    }


def dsh_patch_yaml(ep: ClientEndpoint) -> str:
    """Generates a Cordis patch YAML block for DeepSeek Harness ($DSH_HOME/profiles/web/cordis.patch.yml)."""
    modalities = "[text, image]" if ep.vision_enabled else "[text]"
    reasoning_default = dsh_effort_default(ep.effort)
    lines = [
        "# Add to $DSH_HOME/profiles/web/cordis.patch.yml (default ~/.dsh/profiles/web/cordis.patch.yml)",
        "# Note: if you already have an `llm-pi-ai` entry, merge `basalt-colab` under its `providers:` map.",
        "- id: llm-pi-ai",
        "  config:",
        "    providers:",
        "      basalt-colab:",
        "        apiKeyEnv: BASALT_API_KEY",
        "        api: openai-completions",
        f"        baseURL: {ep.openai_base_url}",
        f"        defaultInput: {modalities}",
        "        compat:",
        "          supportsDeveloperRole: false",
        "          maxTokensField: max_tokens",
        f"        reasoning: {reasoning_default}",
        "        models:",
        f"          - id: {ep.model}",
        f"            input: {modalities}",
        f"            contextWindow: {ep.max_context}",
        "            maxTokens: 32768",
        "            reasoningEfforts:",
        "              off: none",
        "              low: low",
        "              high: high",
        "              max: max",
    ]
    return "\n".join(lines) + "\n"


def claude_settings_dict(ep: ClientEndpoint) -> dict[str, Any]:
    return {
        "env": {
            "ANTHROPIC_BASE_URL": ep.base_url,
            "ANTHROPIC_API_KEY": ep.api_key,
            "ANTHROPIC_AUTH_TOKEN": ep.api_key,
            "ANTHROPIC_MODEL": ep.model,
            "ANTHROPIC_SMALL_FAST_MODEL": ep.model,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
    }


def codex_config_toml(ep: ClientEndpoint) -> str:
    return (
        "# Add to ~/.codex/config.toml (export OPENAI_API_KEY from `basalt-colab env`)\n"
        'model_provider = "basalt-colab"\n'
        f"model = {json.dumps(ep.model)}\n"
        "\n"
        "[model_providers.basalt-colab]\n"
        'name = "Basalt Colab (DeepSeek-V4.1-Flash)"\n'
        f"base_url = {json.dumps(ep.openai_base_url)}\n"
        'env_key = "OPENAI_API_KEY"\n'
        'wire_api = "responses"\n'
    )


def openai_curl_snippet(ep: ClientEndpoint) -> str:
    payload = json.dumps(
        {
            "model": ep.model,
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": False,
        }
    )
    return (
        f"curl -sS {ep.openai_base_url}/chat/completions \\\n"
        f"  -H 'Authorization: Bearer {ep.api_key}' \\\n"
        f"  -H 'Content-Type: application/json' \\\n"
        f"  -d '{payload}'"
    )


def client_bundle(ep: ClientEndpoint, client: str) -> dict[str, Any]:
    c = resolve_client(client)
    if c is None:
        raise ValueError(f"unknown client {client!r} (expected one of {VALID_CLIENTS})")

    env_pairs: tuple[tuple[str, str], ...]
    if c == "dsh":
        env_pairs = (("BASALT_API_KEY", ep.api_key),)
        return {
            "id": "dsh",
            "label": "DeepSeek Harness",
            "subtitle": "OpenAI-compatible `llm-pi-ai` provider with DeepSeek-V4.1 efforts (off/low/high/max) & Dynamic Vision",
            "file_hint": "~/.dsh/profiles/web/cordis.patch.yml (or $DSH_HOME/profiles/<profile>/cordis.patch.yml)",
            "snippet_lang": "yaml",
            "snippet": dsh_patch_yaml(ep),
            "env_sh": format_env_lines(env_pairs, powershell=False),
            "env_ps": format_env_lines(env_pairs, powershell=True),
            "run_hint": f"BASALT_API_KEY={shlex.quote(ep.api_key)} npx @deepseek-ai/dsh web",
            "notes": "Tip: put BASALT_API_KEY=... in ~/.dsh/.env so dsh picks it up automatically; dsh reloads cordis.patch.yml on the next request.",
        }

    if c == "claude":
        env_pairs = (
            ("ANTHROPIC_BASE_URL", ep.base_url),
            ("ANTHROPIC_API_KEY", ep.api_key),
            ("ANTHROPIC_AUTH_TOKEN", ep.api_key),
            ("ANTHROPIC_MODEL", ep.model),
            ("ANTHROPIC_SMALL_FAST_MODEL", ep.model),
            ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1"),
        )
        return {
            "id": "claude",
            "label": "Claude Code",
            "subtitle": "Native Anthropic Messages API (/v1/messages) with thinking & tool calls",
            "file_hint": "~/.claude/settings.json (or export in shell)",
            "snippet_lang": "json",
            "snippet": json.dumps(claude_settings_dict(ep), indent=2) + "\n",
            "env_sh": format_env_lines(env_pairs, powershell=False),
            "env_ps": format_env_lines(env_pairs, powershell=True),
            "run_hint": 'eval "$(basalt-colab env)" && claude',
            "notes": "Paste into ~/.claude/settings.json for persistent config, or run the one-liner in any shell.",
        }

    if c == "codex":
        env_pairs = (
            ("OPENAI_BASE_URL", ep.openai_base_url),
            ("OPENAI_API_KEY", ep.api_key),
            ("OPENAI_MODEL", ep.model),
        )
        return {
            "id": "codex",
            "label": "Codex CLI",
            "subtitle": "Native OpenAI Responses API (/v1/responses)",
            "file_hint": "~/.codex/config.toml",
            "snippet_lang": "toml",
            "snippet": codex_config_toml(ep),
            "env_sh": format_env_lines(env_pairs, powershell=False),
            "env_ps": format_env_lines(env_pairs, powershell=True),
            "run_hint": 'eval "$(basalt-colab env)" && codex',
            "notes": "Add the provider block to ~/.codex/config.toml and export OPENAI_API_KEY before running codex.",
        }

    if c == "opencode":
        doc = opencode_config_dict(ep, key_mode="env")
        env_pairs = (("OPENAI_API_KEY", ep.api_key),)
        return {
            "id": "opencode",
            "label": "OpenCode",
            "subtitle": "OpenAI-compatible provider with interleaved reasoning_content & vision",
            "file_hint": "~/.config/opencode/opencode.json (uses {env:OPENAI_API_KEY}, safe for git)",
            "snippet_lang": "json",
            "snippet": json.dumps(doc, indent=2) + "\n",
            "env_sh": format_env_lines(env_pairs, powershell=False),
            "env_ps": format_env_lines(env_pairs, powershell=True),
            "run_hint": 'eval "$(basalt-colab env)" && opencode',
            "notes": "Or run `basalt-colab connect opencode --write` to update the managed file and launch with OPENCODE_CONFIG=...",
        }

    # Generic OpenAI / Anthropic API (SDKs, Cursor, curl)
    env_pairs = env_vars(ep)
    return {
        "id": "openai",
        "label": "OpenAI / Anthropic API",
        "subtitle": "Generic OpenAI (/v1) & Anthropic (/v1/messages) settings for SDKs, Cursor, and curl",
        "file_hint": "App Provider Settings",
        "snippet_lang": "text",
        "snippet": (
            f"OpenAI Base URL:    {ep.openai_base_url}\n"
            f"Anthropic Base URL: {ep.base_url}\n"
            f"API Key:            {ep.api_key}\n"
            f"Model ID:           {ep.model}\n"
            f"Context Window:     {ep.max_context}\n"
            f"Vision Input:       {'enabled (dynamic)' if ep.vision_enabled else 'disabled'}\n"
            f"\n# Quick curl test:\n{openai_curl_snippet(ep)}\n"
        ),
        "env_sh": format_env_lines(env_pairs, powershell=False),
        "env_ps": format_env_lines(env_pairs, powershell=True),
        "run_hint": 'eval "$(basalt-colab env)"',
        "notes": "Non-strict model routing is enabled by default, so any model name sent to /v1 routes to the active DeepSeek-V4.1-Flash pack.",
    }


def all_client_bundles(ep: ClientEndpoint) -> dict[str, Any]:
    return {
        "endpoint": {
            "base_url": ep.base_url,
            "openai_base_url": ep.openai_base_url,
            "api_key": ep.api_key,
            "model": ep.model,
            "quant": ep.quant,
            "effort": ep.effort,
            "max_context": ep.max_context,
            "vision": ep.vision,
        },
        "clients": [client_bundle(ep, cid) for cid in VALID_CLIENTS],
    }
