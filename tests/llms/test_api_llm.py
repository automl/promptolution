import asyncio
from concurrent.futures import TimeoutError as FuturesTimeout
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from promptolution.llms import APILLM


class _FakeSem:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _make_api_stub(**attrs):
    """Create an APILLM instance via __new__ with provided attributes."""
    api = APILLM.__new__(APILLM)
    api._call_kwargs = {}
    for key, value in attrs.items():
        setattr(api, key, value)
    return api


def test_api_llm_initialization():
    """Test that APILLM initializes correctly."""
    # Create patches for all dependencies
    with patch("promptolution.llms.api_llm.AsyncOpenAI") as mock_client_class, patch(
        "promptolution.llms.api_llm.asyncio"
    ) as mock_asyncio:
        # Configure the mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_semaphore = MagicMock()
        mock_asyncio.Semaphore.return_value = mock_semaphore

        # Create APILLM instance
        api_llm = APILLM(
            api_url="https://api.example.com", model_id="gpt-4", api_key="test-token", max_concurrent_calls=10
        )

        # Verify AsyncOpenAI was called correctly
        mock_client_class.assert_called_once()
        args, kwargs = mock_client_class.call_args
        assert kwargs["base_url"] == "https://api.example.com"
        assert kwargs["api_key"] == "test-token"

        # Verify semaphore was created
        mock_asyncio.Semaphore.assert_called_once_with(10)

        # Verify instance attributes
        assert api_llm.api_url == "https://api.example.com"
        assert api_llm.model_id == "gpt-4"
        assert api_llm.max_concurrent_calls == 10


def test_ainvoke_once_uses_client_and_timeout(monkeypatch):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    api = _make_api_stub(model_id="m", max_tokens=11, call_timeout_s=0.5, _sem=_FakeSem(), client=client)

    out = asyncio.run(api._ainvoke_once("prompt", "system"))

    assert out is response
    assert create.await_count == 1
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "m"
    assert kwargs["messages"][0]["role"] == "system"
    assert kwargs["max_tokens"] == 11


def test_ainvoke_with_retries_recovers(monkeypatch):
    good = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="done"))])
    api = _make_api_stub(max_retries=2, retry_base_delay_s=0)
    api._ainvoke_once = AsyncMock(side_effect=[Exception("fail"), good])

    async def _sleep(_):
        return None

    monkeypatch.setattr("promptolution.llms.api_llm.asyncio.sleep", _sleep)

    out = asyncio.run(api._ainvoke_with_retries("p", "s"))

    assert out == "done"
    assert api._ainvoke_once.await_count == 2


def test_ainvoke_with_retries_exhausts(monkeypatch):
    api = _make_api_stub(max_retries=1, retry_base_delay_s=0)
    api._ainvoke_once = AsyncMock(side_effect=[Exception("boom"), Exception("boom2")])

    async def _sleep(_):
        return None

    monkeypatch.setattr("promptolution.llms.api_llm.asyncio.sleep", _sleep)

    with pytest.raises(Exception) as excinfo:
        asyncio.run(api._ainvoke_with_retries("p", "s"))

    assert "boom2" in str(excinfo.value)
    assert api._ainvoke_once.await_count == 2


def test_aget_batch_success(monkeypatch):
    api = _make_api_stub(gather_timeout_s=1)
    api._ainvoke_with_retries = AsyncMock(side_effect=["a", "b"])
    monkeypatch.setattr("promptolution.llms.api_llm.asyncio.wait_for", asyncio.wait_for)

    outs = asyncio.run(api._aget_batch(["p1", "p2"], ["s1", "s2"]))

    assert outs == ["a", "b"]
    assert api._ainvoke_with_retries.await_count == 2


def test_aget_batch_raises_on_failure(monkeypatch):
    api = _make_api_stub(gather_timeout_s=1)
    api._ainvoke_with_retries = AsyncMock(side_effect=["ok", Exception("boom")])
    monkeypatch.setattr("promptolution.llms.api_llm.asyncio.wait_for", asyncio.wait_for)

    with pytest.raises(RuntimeError):
        asyncio.run(api._aget_batch(["p1", "p2"], ["s1", "s2"]))


def test_get_response_success(monkeypatch):
    api = _make_api_stub(gather_timeout_s=1)
    api._aget_batch = AsyncMock()

    class _Future:
        def __init__(self, value):
            self.value = value
            self.cancelled = False

        def result(self, timeout=None):
            return self.value

        def cancel(self):
            self.cancelled = True

    fut = _Future(["r1", "r2"])
    api._submit = MagicMock(return_value=fut)

    out = api._get_response(["p1", "p2"], ["s1", "s2"])

    assert out == ["r1", "r2"]
    api._submit.assert_called_once()
    assert fut.cancelled is False


def test_get_response_times_out():
    api = _make_api_stub(gather_timeout_s=1)

    class _Future:
        def __init__(self):
            self.cancelled = False

        def result(self, timeout=None):
            raise FuturesTimeout()

        def cancel(self):
            self.cancelled = True

    fut = _Future()
    api._submit = MagicMock(return_value=fut)

    with pytest.raises(TimeoutError):
        api._get_response(["p"], ["s"])

    assert fut.cancelled is True
