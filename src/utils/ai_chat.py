from collections.abc import Iterator
from typing import Any, Optional

from openai import OpenAI


class AiChat:
    """Wrapper for chatting with a local llama.cpp server via its OpenAI-compatible API.

    Start the server first, e.g. ``llama-server -m model.gguf`` (default: http://127.0.0.1:8080).
    The ``model`` name is often ignored by llama.cpp and can be any placeholder string.
    """

    DEFAULT_BASE_URL = 'http://127.0.0.1:8080/v1'

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = 'no-key',
        model: str = 'local',
        system_prompt: Optional[str] = None,
        timeout: Optional[float] = 600.0,
        **client_kwargs: Any,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self._messages: list[dict[str, str]] = []
        if system_prompt:
            self._messages.append({'role': 'system', 'content': system_prompt})

        self._client = OpenAI(
            base_url=base_url.rstrip('/'),
            api_key=api_key,
            timeout=timeout,
            **client_kwargs,
        )

    @property
    def messages(self) -> list[dict[str, str]]:
        """Return a copy of the current conversation history."""
        return list(self._messages)

    def reset(self) -> None:
        """Clear conversation history, keeping the system prompt if configured."""
        self._messages = []
        if self.system_prompt:
            self._messages.append({'role': 'system', 'content': self.system_prompt})

    def list_models(self) -> list[str]:
        """List model ids exposed by the server."""
        return [m.id for m in self._client.models.list().data]

    def chat(
        self,
        user_message: str,
        *,
        stream: bool = False,
        use_history: bool = True,
        **kwargs: Any,
    ) -> str | Iterator[str]:
        """Send a user message and return the assistant reply.

        When ``use_history`` is True (default), messages are appended to the
        session history. When ``stream`` is True, yields text chunks as they arrive.
        """
        request_messages = self._prepare_messages(user_message, use_history)

        if stream:
            return self._stream_reply(request_messages, use_history, **kwargs)

        response = self._client.chat.completions.create(
            model=self.model, messages=request_messages, stream=False, **kwargs
        )
        reply = _extract_content(response.choices[0].message.content)
        if use_history:
            self._messages.append({'role': 'user', 'content': user_message})
            self._messages.append({'role': 'assistant', 'content': reply})
        return reply

    def complete(
        self, messages: list[dict[str, str]], *, stream: bool = False, **kwargs: Any
    ) -> str | Iterator[str]:
        """Call the chat API with a custom message list (does not update history)."""
        if stream:
            return self._stream_reply(messages, use_history=False, **kwargs)

        response = self._client.chat.completions.create(
            model=self.model, messages=messages, stream=False, **kwargs
        )
        return _extract_content(response.choices[0].message.content)

    def _prepare_messages(
        self, user_message: str, use_history: bool
    ) -> list[dict[str, str]]:
        if use_history:
            return [*self._messages, {'role': 'user', 'content': user_message}]

        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': user_message})
        return messages

    def _stream_reply(
        self, messages: list[dict[str, str]], use_history: bool, **kwargs: Any
    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.model, messages=messages, stream=True, **kwargs
        )

        def generate() -> Iterator[str]:
            parts: list[str] = []
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    parts.append(delta)
                    yield delta

            if use_history:
                user_msg = messages[-1]
                if user_msg['role'] == 'user':
                    self._messages.append(user_msg)
                self._messages.append({'role': 'assistant', 'content': ''.join(parts)})

        return generate()


def _extract_content(content: Optional[str]) -> str:
    return content or ''


if __name__ == '__main__':
    # Example usage
    chat = AiChat(system_prompt='You are a helpful assistant.')
    print(chat.chat('Hello, who are you?'))
