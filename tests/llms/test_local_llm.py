from unittest.mock import MagicMock, patch

import pytest

from promptolution.llms import LocalLLM


@pytest.fixture
def mock_local_dependencies():
    """Set up mocks for LocalLLM dependencies."""
    with patch("promptolution.llms.local_llm.pipeline") as mock_pipeline_func, patch(
        "promptolution.llms.local_llm.torch"
    ) as mock_torch:
        # Create a mock pipeline object (not a list!)
        mock_pipeline_obj = MagicMock()

        # Configure the pipeline function to return the pipeline object
        mock_pipeline_func.return_value = mock_pipeline_obj

        # Configure mock tokenizer on the pipeline object
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 50256
        mock_tokenizer.padding_side = None
        mock_pipeline_obj.tokenizer = mock_tokenizer

        # Configure the pipeline object's __call__ method to return responses
        mock_pipeline_obj.return_value = [{"generated_text": "Mock response 1"}, {"generated_text": "Mock response 2"}]

        yield {"pipeline": mock_pipeline_func, "torch": mock_torch, "pipeline_obj": mock_pipeline_obj}


def test_local_llm_initialization(mock_local_dependencies):
    """Test that LocalLLM initializes correctly."""
    # Create LocalLLM instance
    local_llm = LocalLLM(model_id="gpt2", batch_size=4)

    # Verify pipeline was created correctly
    mock_local_dependencies["pipeline"].assert_called_once_with(
        "text-generation",
        model="gpt2",
        model_kwargs={"torch_dtype": mock_local_dependencies["torch"].bfloat16},
        device_map="auto",
        max_new_tokens=256,
        batch_size=4,
        num_return_sequences=1,
        return_full_text=False,
    )

    # Verify tokenizer attributes were set
    assert local_llm.pipeline.tokenizer.pad_token_id == local_llm.pipeline.tokenizer.eos_token_id
    assert local_llm.pipeline.tokenizer.padding_side == "left"


def test_local_llm_get_response(mock_local_dependencies):
    """Test that LocalLLM._get_response works correctly."""
    local_llm = LocalLLM(model_id="gpt2", batch_size=4)

    # Mock prompts
    prompts = ["Hello, world!", "How are you?"]
    sys_prompts = ["System prompt 1", "System prompt 2"]

    # Call _get_response
    responses = local_llm._get_response(prompts, system_prompts=sys_prompts)

    # Verify the responses are as expected
    assert len(responses) == 2
    assert responses[0] == "Mock response 1"
    assert responses[1] == "Mock response 2"


def test_local_llm_get_response_nested_single(mock_local_dependencies):
    """Regression for #73: pipeline can return [[{...}]] for a single prompt; must be flattened."""
    local_llm = LocalLLM(model_id="gpt2", batch_size=1)
    mock_local_dependencies["pipeline_obj"].return_value = [[{"generated_text": "Mock response"}]]

    responses = local_llm._get_response(["Hello, world!"], system_prompts=["System prompt"])

    assert responses == ["Mock response"]
