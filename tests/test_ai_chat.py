import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import cs.ai_chat as ai_chat_mod
from cs.ai_chat import AiChat, _extract_content


class FakeCompletions:
    def __init__(self):
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return [_chunk("Hel"), _chunk("lo"), _chunk(None)]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="world"))]
        )


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeModels:
    def list(self):
        return SimpleNamespace(
            data=[SimpleNamespace(id="model-a"), SimpleNamespace(id="model-b")]
        )


class FakeOpenAIClient:
    instances: list["FakeOpenAIClient"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat = FakeChat()
        self.models = FakeModels()
        FakeOpenAIClient.instances.append(self)


def _chunk(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
    )


@pytest.fixture
def fake_openai(monkeypatch):
    FakeOpenAIClient.instances.clear()
    monkeypatch.setattr(ai_chat_mod, "OpenAI", FakeOpenAIClient)
    return FakeOpenAIClient


def test_init_strips_base_url_and_sets_system_prompt(fake_openai):
    chat = AiChat(
        base_url="http://127.0.0.1:8080/v1/",
        system_prompt="be helpful",
        model="test-model",
    )

    client = fake_openai.instances[0]
    assert client.kwargs["base_url"] == "http://127.0.0.1:8080/v1"
    assert client.kwargs["api_key"] == "no-key"
    assert chat.model == "test-model"
    assert chat.messages == [{"role": "system", "content": "be helpful"}]


def test_reset_keeps_system_prompt(fake_openai):
    chat = AiChat(system_prompt="sys")
    chat._messages.append({"role": "user", "content": "hi"})

    chat.reset()

    assert chat.messages == [{"role": "system", "content": "sys"}]


def test_chat_non_stream_with_history(fake_openai):
    chat = AiChat(model="m1")

    reply = chat.chat("hello")

    assert reply == "world"
    assert chat.messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]

    client = fake_openai.instances[0]
    call = client.chat.completions.calls[-1]
    assert call["model"] == "m1"
    assert call["stream"] is False
    assert call["messages"] == [{"role": "user", "content": "hello"}]


def test_chat_without_history(fake_openai):
    chat = AiChat(system_prompt="sys")
    chat.chat("first")
    reply = chat.chat("second", use_history=False)

    assert reply == "world"
    assert len(chat.messages) == 3
    assert chat.messages[1]["content"] == "first"
    assert chat.messages[2]["content"] == "world"

    call = fake_openai.instances[0].chat.completions.calls[-1]
    assert call["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "second"},
    ]


def test_chat_stream_updates_history(fake_openai):
    chat = AiChat()

    chunks = list(chat.chat("stream me", stream=True))

    assert chunks == ["Hel", "lo"]
    assert chat.messages == [
        {"role": "user", "content": "stream me"},
        {"role": "assistant", "content": "Hello"},
    ]

    call = fake_openai.instances[0].chat.completions.calls[-1]
    assert call["stream"] is True


def test_complete_does_not_update_history(fake_openai):
    chat = AiChat(system_prompt="sys")
    chat._messages.append({"role": "user", "content": "old"})

    reply = chat.complete([{"role": "user", "content": "new"}])

    assert reply == "world"
    assert len(chat.messages) == 2
    assert chat.messages[-1]["content"] == "old"


def test_complete_stream(fake_openai):
    chat = AiChat()

    chunks = list(chat.complete([{"role": "user", "content": "x"}], stream=True))

    assert chunks == ["Hel", "lo"]
    assert chat.messages == []


def test_list_models(fake_openai):
    chat = AiChat()

    assert chat.list_models() == ["model-a", "model-b"]


def test_extract_content_handles_none():
    assert _extract_content(None) == ""
    assert _extract_content("ok") == "ok"
