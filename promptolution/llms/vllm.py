"""Module for running language models locally using the vLLM library."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.utils.config import ExperimentConfig
    from transformers import PreTrainedTokenizer


from promptolution.llms.base_llm import BaseLLM
from promptolution.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams

    imports_successful = True
except ImportError:
    imports_successful = False


class VLLM(BaseLLM):
    """A class for running language models using the vLLM library.

    This class sets up a vLLM inference engine with specified model parameters
    and provides a method to generate responses for given prompts.

    Attributes:
        llm (vllm.LLM): The vLLM inference engine.
        tokenizer (PreTrainedTokenizer): The tokenizer for the model.
        sampling_params (vllm.SamplingParams): Parameters for text generation.

    Methods:
        get_response: Generate responses for a list of prompts.
        update_token_count: Update the token count based on the given inputs and outputs.
    """

    tokenizer: "PreTrainedTokenizer"

    def __init__(
        self,
        model_id: str,
        batch_size: Optional[int] = None,
        max_generated_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9,
        model_storage_path: Optional[str] = None,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 2048,
        trust_remote_code: bool = False,
        seed: int = 42,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        config: Optional["ExperimentConfig"] = None,
    ) -> None:
        """Initialize the VLLM with a specific model.

        Args:
            model_id (str): The identifier of the model to use.
            batch_size (int, optional): The batch size for text generation. Defaults to 8.
            max_generated_tokens (int, optional): Maximum number of tokens to generate. Defaults to 256.
            temperature (float, optional): Sampling temperature. Defaults to 0.1.
            top_p (float, optional): Top-p sampling parameter. Defaults to 0.9.
            model_storage_path (str, optional): Directory to store the model. Defaults to None.
            dtype (str, optional): Data type for model weights. Defaults to "float16".
            tensor_parallel_size (int, optional): Number of GPUs for tensor parallelism. Defaults to 1.
            gpu_memory_utilization (float, optional): Fraction of GPU memory to use. Defaults to 0.95.
            max_model_len (int, optional): Maximum sequence length for the model. Defaults to 2048.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
            seed (int, optional): Random seed for the model. Defaults to 42.
            llm_kwargs (dict, optional): Additional keyword arguments for the LLM. Defaults to None.
            config (ExperimentConfig, optional): Configuration for the LLM, overriding defaults.

        Note:
            This method sets up a vLLM engine with specified parameters for efficient inference.
        """
        if not imports_successful:
            raise ImportError(
                "Could not import at least one of the required libraries: transformers, vllm. "
                "Please ensure they are installed in your environment."
            )

        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code

        super().__init__(config)

        # Configure sampling parameters
        self.sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_generated_tokens, seed=seed
        )

        llm_kwargs = llm_kwargs or {}
        # Initialize the vLLM engine with both explicit parameters and any additional kwargs
        llm_params: Dict[str, Any] = {
            "model": model_id,
            "tokenizer": model_id,
            "dtype": self.dtype,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "download_dir": model_storage_path,
            "trust_remote_code": self.trust_remote_code,
            "seed": seed,
            **llm_kwargs,
        }

        self.llm = LLM(**llm_params)

        self.tokenizer = self.llm.get_tokenizer()

        if batch_size is None:
            cache_config = self.llm.llm_engine.model_executor.cache_config
            if (
                cache_config.num_gpu_blocks is not None
                and cache_config.block_size is not None
                and self.max_model_len is not None
            ):
                self.batch_size = int(
                    (cache_config.num_gpu_blocks * cache_config.block_size / self.max_model_len) * 0.95
                )
                logger.info(f"🚀 Batch size set to {self.batch_size} based on GPU memory.")
            else:
                self.batch_size = 1
                logger.warning("⚠️ Could not determine batch size from GPU memory. Using batch size of 1.")
        else:
            self.batch_size = batch_size

    def _get_response(self, prompts: List[str], system_prompts: List[str]) -> List[str]:
        """Generate responses for a list of prompts using the vLLM engine.

        Args:
            prompts (list[str]): A list of input prompts.
            system_prompts (list[str]): A list of system prompts to guide the model's behavior.

        Returns:
            list[str]: A list of generated responses corresponding to the input prompts.

        Note:
            This method uses vLLM's batched generation capabilities for efficient inference.
            It also counts input and output tokens.
        """
        prompts = [
            str(
                self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": sys_prompt,
                        },
                        {"role": "user", "content": prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            for prompt, sys_prompt in zip(prompts, system_prompts)
        ]

        # generate responses for self.batch_size prompts at the same time
        all_responses = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            outputs = self.llm.generate(batch, self.sampling_params)
            responses = [output.outputs[0].text for output in outputs]

            all_responses.extend(responses)

        return all_responses

    def update_token_count(self, inputs: List[str], outputs: List[str]) -> None:
        """Update the token count based on the given inputs and outputs.

            Uses the tokenizer to count the tokens.

        Args:
            inputs (List[str]): A list of input prompts.
            outputs (List[str]): A list of generated responses.
        """
        for input in inputs:
            self.input_token_count += len(self.tokenizer.encode(input))

        for output in outputs:
            self.output_token_count += len(self.tokenizer.encode(output))

    def set_generation_seed(self, seed: int) -> None:
        """Set the random seed for text generation.

        Args:
            seed (int): Random seed for text generation.
        """
        self.sampling_params.seed = seed
