from unittest.mock import MagicMock, patch

import pytest
from transformers import AutoTokenizer

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


@pytest.mark.parametrize(
    "model_id",
    [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "microsoft/Phi-3.5-mini-instruct",
        "mistralai/Mistral-Nemo-Instruct-2407",
    ],
)
def test_local_llm_chat_template_renders(model_id):
    """Regression for #71: message dicts must use 'content' key so the
    tokenizer's chat template renders the system and user text."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    with patch("promptolution.llms.local_llm.pipeline") as mock_pipeline_func, patch(
        "promptolution.llms.local_llm.torch"
    ):
        mock_pipeline_obj = MagicMock()
        mock_pipeline_obj.tokenizer = tokenizer
        mock_pipeline_func.return_value = mock_pipeline_obj

        def fake_call(inputs, **_):
            return [
                [{"generated_text": tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)}]
                for msg in inputs
            ]

        mock_pipeline_obj.side_effect = fake_call

        local_llm = LocalLLM(model_id=model_id, batch_size=2)
        prompts = ["What is 2 + 2?", "Name a colour."]
        sys_prompts = ["You are a math tutor.", "You are concise."]

        responses = local_llm._get_response(prompts, system_prompts=sys_prompts)

        assert len(responses) == 2
        for response, prompt, sys_prompt in zip(responses, prompts, sys_prompts):
            assert prompt in response
            assert sys_prompt in response
