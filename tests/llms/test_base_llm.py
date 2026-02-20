from tests.mocks.dummy_config import DummyConfig
from tests.mocks.mock_llm import MockLLM


def test_base_llm_token_count_and_reset():
    llm = MockLLM()
    llm.update_token_count(["a b"], ["c d e"])
    counts = llm.get_token_count()
    assert counts["input_tokens"] == 2
    assert counts["output_tokens"] == 3

    llm.reset_token_count()
    assert llm.get_token_count()["total_tokens"] == 0


def test_base_llm_default_and_list_system_prompts():
    llm = MockLLM()
    res_single = llm.get_response("hello")
    assert res_single == ["Mock response for: hello"]

    res_multi = llm.get_response(["p1", "p2"], system_prompts=["s1", "s2"])
    assert res_multi == ["Mock response for: p1", "Mock response for: p2"]


def test_base_llm_config_applied():
    cfg = DummyConfig()
    llm = MockLLM(predetermined_responses=["r1"], add_prompt_tags=False, config=cfg)
    assert cfg.applied is True
    assert getattr(llm, "applied") is True


def test_base_llm_set_generation_seed():
    llm = MockLLM()
    llm.set_generation_seed(123)
    assert llm._generation_seed == 123
