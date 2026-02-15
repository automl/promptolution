from unittest.mock import MagicMock, patch

import pytest

from promptolution.llms.vllm import VLLM

vllm = pytest.importorskip("vllm")
transformers = pytest.importorskip("transformers")


@pytest.fixture
def mock_vllm_dependencies():
    """Set up comprehensive mocks for VLLM dependencies."""
    with patch("promptolution.llms.vllm.LLM") as mock_llm_class, patch(
        "promptolution.llms.vllm.SamplingParams"
    ) as mock_sampling_params, patch("promptolution.llms.vllm.AutoTokenizer.from_pretrained") as mock_from_pretrained:
        # --- LLM and Engine Mock Setup ---
        mock_llm_instance = MagicMock()
        mock_cache_config = MagicMock(num_gpu_blocks=100, block_size=16)
        mock_executor = MagicMock(cache_config=mock_cache_config)
        mock_engine = MagicMock(model_executor=mock_executor)
        mock_llm_instance.llm_engine = mock_engine
        mock_llm_class.return_value = mock_llm_instance

        def mock_generate_side_effect(prompts_list, *args, **kwargs):
            return [
                MagicMock(outputs=[MagicMock(text=f"Mocked response for prompt {i}")])
                for i, _ in enumerate(prompts_list)
            ]

        mock_llm_instance.generate.side_effect = mock_generate_side_effect

        # --- Tokenizer Mock Setup (The Fix) ---
        # 1. Create the mock object we want to be our tokenizer.
        mock_tokenizer_instance = MagicMock()

        # 2. Configure its methods directly.
        mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer_instance.apply_chat_template.return_value = "<mocked_chat_template>"

        # 3. Tell the patch for "from_pretrained" to return our configured instance.
        # This is the most critical change.
        mock_from_pretrained.return_value = mock_tokenizer_instance

        # 4. Make sure llm_instance.get_tokenizer() returns the mock tokenizer
        mock_llm_instance.get_tokenizer.return_value = mock_tokenizer_instance

        # --- Sampling Params Mock Setup ---
        mock_sampling_params_instance = MagicMock()
        mock_sampling_params.return_value = mock_sampling_params_instance

        yield {
            "llm_class": mock_llm_class,
            "llm_instance": mock_llm_instance,
            "tokenizer": mock_tokenizer_instance,
            "sampling_params_class": mock_sampling_params,
            "sampling_params_instance": mock_sampling_params_instance,
        }


def test_vllm_get_response(mock_vllm_dependencies):
    """Test that VLLM._get_response works correctly with explicit batch_size."""
    # Create VLLM instance with explicit batch_size to avoid calculation
    vllm_instance = VLLM(model_id="mock-model", batch_size=4)

    # Verify the mocks were used
    mock_vllm_dependencies["llm_class"].assert_called_once()

    # Call get_response
    prompts = ["Test prompt 1", "Test prompt 2"]
    system_prompts = ["Be helpful", "Be concise"]
    responses = vllm_instance._get_response(prompts, system_prompts)

    # Verify tokenizer was used correctly
    assert mock_vllm_dependencies["tokenizer"].apply_chat_template.call_count == 2

    # Verify LLM generate was called
    mock_vllm_dependencies["llm_instance"].generate.assert_called_once()

    # Verify responses
    assert len(responses) == 2
    assert responses[0] == "Mocked response for prompt 0"
    assert responses[1] == "Mocked response for prompt 1"


def test_vllm_with_auto_batch_size(mock_vllm_dependencies):
    """Test VLLM with automatic batch size calculation."""
    # Set up cache config for batch size calculation
    mock_vllm_dependencies["llm_instance"].llm_engine.model_executor.cache_config.num_gpu_blocks = 1000
    mock_vllm_dependencies["llm_instance"].llm_engine.model_executor.cache_config.block_size = 16

    # Create VLLM instance with batch_size=None to trigger auto calculation
    # With max_num_batched_tokens=16384 and max_model_len=2048:
    # token_limited = 16384 // 2048 = 8
    # batch_size = min(max_num_seqs=10, token_limited=8) = 8
    vllm_instance = VLLM(
        model_id="mock-model",
        batch_size=None,
        max_model_len=2048,
        llm_kwargs={"max_num_seqs": 10, "max_num_batched_tokens": 16384},
    )

    # Verify batch_size is greater than zero
    assert vllm_instance.batch_size > 0, "Batch size should be greater than zero"
    # With max_num_batched_tokens=16384, max_model_len=2048, max_num_seqs=10
    # token_limited = 16384 // 2048 = 8
    # batch_size = min(10, 8) = 8
    assert vllm_instance.batch_size == 8, f"Expected batch_size=8, got {vllm_instance.batch_size}"

    # Test with a single prompt
    prompts = ["Test prompt"]
    system_prompts = ["Be helpful"]
    responses = vllm_instance._get_response(prompts, system_prompts)

    # Verify we get exactly one response for one prompt
    assert len(responses) == 1
    assert responses[0] == "Mocked response for prompt 0"


def test_vllm_initialization_parameters(mock_vllm_dependencies):
    """Test that VLLM correctly passes parameters to underlying LLM."""
    # Create VLLM instance with custom parameters
    vllm_instance = VLLM(
        model_id="mock-model",
        batch_size=8,
        max_generated_tokens=512,
        temperature=0.5,
        top_p=0.95,
        dtype="float16",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        trust_remote_code=True,
        seed=123,
        llm_kwargs={"custom_param": "value"},
    )

    # Verify LLM was initialized with correct parameters
    call_args = mock_vllm_dependencies["llm_class"].call_args
    assert call_args[1]["model"] == "mock-model"
    assert call_args[1]["tokenizer"] == "mock-model"
    assert call_args[1]["dtype"] == "float16"
    assert call_args[1]["tensor_parallel_size"] == 2
    assert call_args[1]["gpu_memory_utilization"] == 0.9
    assert call_args[1]["max_model_len"] == 4096
    assert call_args[1]["trust_remote_code"] is True
    assert call_args[1]["seed"] == 123
    assert call_args[1]["custom_param"] == "value"

    # Verify SamplingParams was initialized with correct parameters
    call_args = mock_vllm_dependencies["sampling_params_class"].call_args
    assert call_args[1]["temperature"] == 0.5
    assert call_args[1]["top_p"] == 0.95
    assert call_args[1]["max_tokens"] == 512
    assert call_args[1]["seed"] == 123

    # Verify batch_size was set correctly
    assert vllm_instance.batch_size == 8
