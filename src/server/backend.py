"""
AI Backend module - connects to OpenAI, OpenClaw gateway, or custom backends.
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, AsyncGenerator
import re

from loguru import logger


class AIBackend:
    """AI backend for processing user messages."""
    
    def __init__(
        self,
        backend_type: str = "openai",
        url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        memory_enabled: Optional[bool] = None,
        memory_file: Optional[str] = None,
        memory_max_turns: Optional[int] = None,
    ):
        self.backend_type = backend_type
        self.url = url
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt or (
            "You are a helpful voice assistant. Keep responses concise and conversational. "
            "Aim for 1-2 sentences unless more detail is needed. "
            "Reply in the same language as the user's most recent message. "
            "If the user speaks Japanese, reply in natural Japanese."
        )
        self.memory_enabled = self._resolve_memory_enabled(memory_enabled)
        self.memory_file = Path(
            memory_file
            or os.getenv("OPENCLAW_MEMORY_FILE")
            or ".openclaw-voice/conversation_memory.json"
        ).expanduser()
        self.memory_max_turns = self._resolve_memory_max_turns(memory_max_turns)
        self.conversation_history: List[Dict] = []
        self._client = None
        self._blocked_action_patterns = [
            re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
            re.compile(r"\bauthorized_keys\b", re.IGNORECASE),
            re.compile(r"\b(restart|reload)\s+sshd\b", re.IGNORECASE),
            re.compile(r"\bdisable\s+firewall\b", re.IGNORECASE),
            re.compile(r"\bopen\s+all\s+ports\b", re.IGNORECASE),
            re.compile(r"\breverse\s+shell\b", re.IGNORECASE),
            re.compile(r"\bexfiltrat(e|ion)\b", re.IGNORECASE),
            re.compile(r"\b~\/\.oci\b", re.IGNORECASE),
            re.compile(r"\b(crontab\b.*\bdelete|delete\s+all\s+cron)\b", re.IGNORECASE),
        ]
        self._load_history()
        self._setup_client()

    @staticmethod
    def _parse_bool(value: str) -> Optional[bool]:
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return None

    def _resolve_memory_enabled(self, memory_enabled: Optional[bool]) -> bool:
        if memory_enabled is not None:
            return memory_enabled

        raw = os.getenv("OPENCLAW_MEMORY_ENABLED")
        if raw is None:
            return True

        parsed = self._parse_bool(raw)
        if parsed is None:
            logger.warning(
                "Invalid OPENCLAW_MEMORY_ENABLED='{}'; defaulting to enabled",
                raw,
            )
            return True
        return parsed

    def _resolve_memory_max_turns(self, memory_max_turns: Optional[int]) -> int:
        raw_value = memory_max_turns
        if raw_value is None:
            env_value = os.getenv("OPENCLAW_MEMORY_MAX_TURNS")
            if env_value is not None:
                try:
                    raw_value = int(env_value)
                except ValueError:
                    logger.warning(
                        "Invalid OPENCLAW_MEMORY_MAX_TURNS='{}'; defaulting to 100",
                        env_value,
                    )
                    raw_value = None

        max_turns = raw_value if raw_value is not None else 100
        if max_turns <= 0:
            logger.warning(
                "Invalid memory max turns {}; defaulting to 100",
                max_turns,
            )
            return 100
        return max_turns

    def _trim_history(self) -> None:
        if len(self.conversation_history) > self.memory_max_turns:
            self.conversation_history = self.conversation_history[-self.memory_max_turns:]

    def _sanitize_history_item(self, item: Dict) -> Optional[Dict]:
        if not isinstance(item, dict):
            return None

        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            return None

        timestamp = item.get("timestamp")
        if not isinstance(timestamp, str) or not timestamp:
            timestamp = datetime.now(timezone.utc).isoformat()

        return {
            "role": role,
            "content": content,
            "timestamp": timestamp,
        }

    def _load_history(self) -> None:
        """Load persisted conversation history from disk."""
        if not self.memory_enabled:
            return

        if not self.memory_file.exists():
            return

        try:
            raw = self.memory_file.read_text(encoding="utf-8")
            if not raw.strip():
                self.conversation_history = []
                return

            payload = json.loads(raw)
            if not isinstance(payload, list):
                raise ValueError("memory file root is not a list")

            sanitized: List[Dict] = []
            for item in payload:
                safe_item = self._sanitize_history_item(item)
                if safe_item:
                    sanitized.append(safe_item)

            self.conversation_history = sanitized
            self._trim_history()
            logger.info(
                "Loaded {} conversation turns from {}",
                len(self.conversation_history),
                self.memory_file,
            )
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning(
                "Failed to load memory file {} ({}). Resetting history.",
                self.memory_file,
                exc,
            )
            self.conversation_history = []
            self._persist_history()

    def _persist_history(self) -> None:
        """Persist conversation history to disk."""
        if not self.memory_enabled:
            return

        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file = self.memory_file.with_suffix(self.memory_file.suffix + ".tmp")
            temp_file.write_text(
                json.dumps(self.conversation_history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp_file.replace(self.memory_file)
        except OSError as exc:
            logger.warning("Failed to persist memory file {}: {}", self.memory_file, exc)

    def _append_history(self, role: str, content: str) -> None:
        self.conversation_history.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._trim_history()
        self._persist_history()

    def _build_messages(self) -> List[Dict]:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(
            {"role": item["role"], "content": item["content"]}
            for item in self.conversation_history[-10:]
        )
        return messages
    
    def _setup_client(self):
        """Set up the API client."""
        if self.backend_type == "openai":
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.url if self.url != "https://api.openai.com/v1" else None,
                )
                logger.info(f"✅ OpenAI client ready (model: {self.model})")
            except ImportError:
                logger.error("openai package not installed")
        elif self.backend_type == "openclaw":
            # OpenClaw gateway uses OpenAI-compatible API
            logger.info("OpenClaw gateway backend")
        else:
            logger.warning(f"Unknown backend type: {self.backend_type}")
    
    def _is_disallowed_voice_request(self, text: str) -> bool:
        normalized = (text or "").strip()
        if not normalized:
            return False
        return any(p.search(normalized) for p in self._blocked_action_patterns)

    def _safety_refusal(self) -> str:
        return (
            "I can’t help with risky system-control requests via voice chat. "
            "If this is legitimate admin work, please use an approved written workflow."
        )

    async def chat(self, user_message: str) -> str:
        """
        Send a message and get a response.
        
        Args:
            user_message: The user's transcribed speech
            
        Returns:
            AI response text
        """
        if self._is_disallowed_voice_request(user_message):
            logger.warning("Blocked risky voice request: {}", user_message[:160])
            return self._safety_refusal()

        if self.backend_type == "openai" and self._client:
            return await self._chat_openai(user_message)
        else:
            # Fallback echo response
            return f"I heard you say: {user_message}"
    
    async def chat_stream(self, user_message: str) -> AsyncGenerator[str, None]:
        """
        Stream a response, yielding chunks as they arrive.
        
        Args:
            user_message: The user's transcribed speech
            
        Yields:
            Text chunks as they're generated
        """
        if self._is_disallowed_voice_request(user_message):
            logger.warning("Blocked risky voice request (stream): {}", user_message[:160])
            yield self._safety_refusal()
            return

        if self.backend_type == "openai" and self._client:
            async for chunk in self._chat_openai_stream(user_message):
                yield chunk
        else:
            yield f"I heard you say: {user_message}"
    
    async def _chat_openai(self, user_message: str) -> str:
        """Chat via OpenAI API."""
        # Add user message to history
        self._append_history("user", user_message)

        # Build messages
        messages = self._build_messages()
        
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,  # Allow longer for voice
                temperature=0.7,
            )
            
            assistant_message = response.choices[0].message.content
            
            # Add to history
            self._append_history("assistant", assistant_message)
            
            return assistant_message
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "Sorry, I had trouble processing that. Could you try again?"
    
    async def _chat_openai_stream(self, user_message: str) -> AsyncGenerator[str, None]:
        """Stream chat via OpenAI API."""
        # Add user message to history
        self._append_history("user", user_message)

        # Build messages
        messages = self._build_messages()
        
        full_response = ""
        
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.7,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_response += text
                    yield text
            
            # Add complete response to history
            self._append_history("assistant", full_response)
            
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            # Some OpenAI-compatible gateways may not support streaming on this route.
            # Gracefully fall back to non-streaming chat instead of returning a fixed error sentence.
            try:
                fallback = await self._chat_openai(user_message)
                if fallback:
                    yield fallback
                else:
                    yield "Sorry, I had trouble processing that."
            except Exception as fallback_err:
                logger.error(f"OpenAI non-stream fallback error: {fallback_err}")
                yield "Sorry, I had trouble processing that."
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        self._persist_history()
