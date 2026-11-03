from collections.abc import Iterator
from typing import Any, Optional

from openai import OpenAI


class AiChat:
    """通过 OpenAI 兼容接口与本地 llama.cpp 服务对话的封装。

    需先启动服务，例如 ``llama-server -m model.gguf``（默认 http://127.0.0.1:8080）。
    llama.cpp 通常忽略 ``model`` 名，可以随便填一个占位字符串。
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
        """返回当前对话历史的副本。"""
        return list(self._messages)

    def reset(self) -> None:
        """清空对话历史；若配置了系统提示词则保留。"""
        self._messages = []
        if self.system_prompt:
            self._messages.append({'role': 'system', 'content': self.system_prompt})

    def list_models(self) -> list[str]:
        """列出服务端暴露的模型 ID。"""
        return [m.id for m in self._client.models.list().data]

    def chat(
        self,
        user_message: str,
        *,
        stream: bool = False,
        use_history: bool = True,
        **kwargs: Any,
    ) -> str | Iterator[str]:
        """发送用户消息并返回助手回复。

        ``use_history`` 为 True（默认）时，消息会追加到会话历史；
        ``stream`` 为 True 时，边生成边产出文本片段。
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
        """用自定义消息列表调用对话接口（不写入会话历史）。"""
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
    # 使用示例
    chat = AiChat(system_prompt='You are a helpful assistant.')
    print(chat.chat('Hello, who are you?'))
