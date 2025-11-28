"""Module to interface with various language models through their respective APIs."""

import asyncio
import threading
from concurrent.futures import TimeoutError as FuturesTimeout

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from typing import Any, Dict, List, Optional

from promptolution.llms.base_llm import BaseLLM
from promptolution.utils.config import ExperimentConfig
from promptolution.utils.logging import get_logger

logger = get_logger(__name__)


class APILLM(BaseLLM):
    """Persistent asynchronous LLM wrapper using a background event loop."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        max_concurrent_calls: int = 32,
        max_tokens: int = 512,
        call_timeout_s: float = 30.0,  # per request
        gather_timeout_s: float = 120.0,  # whole batch
        max_retries: int = 2,
        retry_base_delay_s: float = 0.5,
        client_kwargs: Optional[Dict[str, Any]] = None,
        call_kwargs: Optional[Dict[str, Any]] = None,
        config: Optional["ExperimentConfig"] = None,
    ) -> None:
        """Initialize the APILLM.

        Args:
            api_url (Optional[str]): Base URL for the API endpoint.
            model_id (Optional[str]): Identifier of the model to call. Must be set.
            api_key (Optional[str]): API key/token for authentication.
            max_concurrent_calls (int): Maximum number of concurrent API calls.
            max_tokens (int): Default maximum number of tokens in model responses.
            call_timeout_s (float): Per-call timeout in seconds.
            gather_timeout_s (float): Timeout in seconds for the entire batch.
            max_retries (int): Number of retry attempts per prompt in addition to the initial call.
            retry_base_delay_s (float): Base delay in seconds for exponential backoff between retries.
            client_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments passed to `AsyncOpenAI(...)`.
            call_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments passed to `client.chat.completions.create(...)`.
            config (Optional[ExperimentConfig]): Configuration for the LLM, overriding defaults.
        """
        self.api_url = api_url
        self.model_id = model_id
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.call_timeout_s = call_timeout_s
        self.gather_timeout_s = gather_timeout_s
        self.max_retries = max_retries
        self.retry_base_delay_s = retry_base_delay_s

        # extra kwargs
        self._client_kwargs: Dict[str, Any] = dict(client_kwargs or {})
        self._call_kwargs: Dict[str, Any] = dict(call_kwargs or {})

        self.max_concurrent_calls = max_concurrent_calls
        super().__init__(config=config)

        # --- persistent loop + semaphore ---
        self._loop = asyncio.new_event_loop()
        self._sem = asyncio.Semaphore(self.max_concurrent_calls)

        def _run_loop() -> None:
            """Run the background event loop forever."""
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._thread = threading.Thread(target=_run_loop, name="APILLMLoop", daemon=True)
        self._thread.start()

        # Create client once; can still be customised via client_kwargs.
        self.client = AsyncOpenAI(
            base_url=self.api_url,
            api_key=self.api_key,
            timeout=self.call_timeout_s,
            **self._client_kwargs,
        )

    # ---------- async bits that run inside the loop ----------
    async def _ainvoke_once(self, prompt: str, system_prompt: str) -> ChatCompletion:
        """Perform a single API call with a per-call timeout.

        Args:
            prompt (str): User prompt content.
            system_prompt (str): System-level instructions for the model.

        Returns:
            ChatCompletion: Raw completion response from the API.

        Raises:
            asyncio.TimeoutError: If the call exceeds `call_timeout_s`.
            Exception: Any exception raised by the underlying client call.
        """
        messages = [
            {"role": "system", "content": str(system_prompt)},
            {"role": "user", "content": str(prompt)},
        ]

        # base kwargs; user can override via call_kwargs
        kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        kwargs.update(self._call_kwargs)

        async with self._sem:
            # per-call timeout enforces failure instead of hang
            return await asyncio.wait_for(
                self.client.chat.completions.create(**kwargs),
                timeout=self.call_timeout_s,
            )

    async def _ainvoke_with_retries(self, prompt: str, system_prompt: str) -> str:
        """Invoke the model with retries and exponential backoff.

        Args:
            prompt (str): User prompt content.
            system_prompt (str): System-level instructions for the model.

        Returns:
            str: The message content of the first choice in the completion.

        Raises:
            Exception: The last exception encountered after all retries are exhausted.
        """
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                r = await self._ainvoke_once(prompt, system_prompt)
                content = r.choices[0].message.content
                if content is None:
                    raise RuntimeError("Empty content from model")
                return content
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    delay = self.retry_base_delay_s * (2**attempt)
                    logger.warning(
                        "LLM call failed (%d/%d): %s — retrying in %.2fs",
                        attempt + 1,
                        self.max_retries + 1,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
        assert last_err is not None
        raise last_err

    async def _aget_batch(self, prompts: List[str], system_prompts: List[str]) -> List[str]:
        """Execute a batch of prompts concurrently and collect responses.

        Args:
            prompts (List[str]): List of user prompts.
            system_prompts (List[str]): List of system prompts; must match `prompts` in length.

        Returns:
            List[str]: List of model outputs. For failed entries, an empty string is inserted.

        Raises:
            TimeoutError: If the entire batch exceeds `gather_timeout_s`.
            RuntimeError: If any of the tasks fails; the first exception is propagated.
        """
        tasks = [asyncio.create_task(self._ainvoke_with_retries(p, s)) for p, s in zip(prompts, system_prompts)]

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.gather_timeout_s,
            )
        except asyncio.TimeoutError:
            for t in tasks:
                t.cancel()
            raise TimeoutError(f"LLM batch timed out after {self.gather_timeout_s}s")

        outs: List[str] = []
        first_exc: Optional[BaseException] = None
        for r in results:
            if isinstance(r, BaseException):
                if first_exc is None:
                    first_exc = r
                outs.append("")
            else:
                outs.append(r)

        if first_exc:
            for t in tasks:
                if not t.done():
                    t.cancel()
            raise RuntimeError(f"LLM batch failed: {first_exc}") from first_exc

        return outs

    # ---------- sync API used by the threads ----------
    def _submit(self, coro):
        """Submit a coroutine to the background event loop.

        Args:
            coro: Coroutine object to be scheduled on the loop.

        Returns:
            concurrent.futures.Future: Future representing the coroutine result.
        """
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _get_response(self, prompts: List[str], system_prompts: List[str]) -> List[str]:
        """Synchronously obtain responses for a batch of prompts.

        This is the main entrypoint used by external callers. It handles system
        prompt broadcasting and delegates the actual work to the async batch
        execution on the background loop.

        Args:
            prompts (List[str]): List of user prompts.
            system_prompts (List[str]): List of system prompts. If a single system
                prompt is provided and multiple prompts are given, the system
                prompt is broadcast to all prompts. Otherwise, the list is
                normalized to match the length of `prompts`.

        Returns:
            List[str]: List of model responses corresponding to `prompts`.

        Raises:
            TimeoutError: If waiting on the batch future exceeds `gather_timeout_s + 5.0`.
            Exception: Any underlying error from the async batch execution.
        """
        fut = self._submit(self._aget_batch(prompts, system_prompts))
        try:
            r = fut.result(timeout=self.gather_timeout_s + 5.0)
            return r
        except FuturesTimeout:
            fut.cancel()
            raise TimeoutError(f"LLM batch (future) timed out after {self.gather_timeout_s + 5.0}s")
        except Exception:
            raise
