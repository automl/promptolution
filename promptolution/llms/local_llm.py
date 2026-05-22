"""Module for running LLMs locally using the Hugging Face Transformers library."""

try:
    import torch
    from transformers import Pipeline, pipeline

    imports_successful = True
except ImportError:
    imports_successful = False

from typing import TYPE_CHECKING, Dict, List, Optional

from promptolution.llms.base_llm import BaseLLM

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.utils.config import ExperimentConfig


class LocalLLM(BaseLLM):
    """A class for running language models locally using the Hugging Face Transformers library.

    This class sets up a text generation pipeline with specified model parameters
    and provides a method to generate responses for given prompts.

    Attributes:
        pipeline (transformers.Pipeline): The text generation pipeline.

    Methods:
        get_response: Generate responses for a list of prompts.
    """

    def __init__(self, model_id: str, batch_size: int = 8, config: Optional["ExperimentConfig"] = None) -> None:
        """Initialize the LocalLLM with a specific model.

        Args:
            model_id (str): The identifier of the model to use (e.g., "gpt2", "facebook/opt-1.3b").
            batch_size (int, optional): The batch size for text generation. Defaults to 8.
            config (ExperimentConfig, optional): "ExperimentConfig" overwriting defaults.

        Note:
            This method sets up a text generation pipeline with bfloat16 precision,
            automatic device mapping, and specific generation parameters.
        """
        if not imports_successful:
            raise ImportError(
                "Could not import at least one of the required libraries: torch, transformers. "
                "Please ensure they are installed in your environment."
            )
        self.pipeline: Pipeline = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            max_new_tokens=256,
            batch_size=batch_size,
            num_return_sequences=1,
            return_full_text=False,
        )
        super().__init__(config)
        self.tokenizer = self.pipeline.tokenizer
        assert self.tokenizer is not None, "Tokenizer must be initialized."
        self.eos_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token_id = self.eos_token_id
        self.tokenizer.padding_side = "left"

    def _get_response(self, prompts: List[str], system_prompts: List[str]) -> List[str]:
        """Generate responses for a list of prompts using the local language model.

        Args:
            prompts (list[str]): A list of input prompts.
            system_prompts (list[str]): A list of system prompts to guide the model's behavior.

        Returns:
            list[str]: A list of generated responses corresponding to the input prompts.

        Note:
            This method uses torch.no_grad() for inference to reduce memory usage.
            It handles both single and batch inputs, ensuring consistent output format.
        """
        inputs: List[List[Dict[str, str]]] = []
        for prompt, sys_prompt in zip(prompts, system_prompts):
            inputs.append([{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}])

        with torch.no_grad():
            response = self.pipeline(inputs, pad_token_id=self.eos_token_id)

        response = [r[0] if isinstance(r, list) else r for r in response]
        response = [r["generated_text"] for r in response]
        return response

    def __del__(self) -> None:
        """Cleanup method to delete the pipeline and free up GPU memory."""
        if hasattr(self, "pipeline"):
            del self.pipeline
        if "torch" in globals() and hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
