import numpy as np

from tests.mocks.dummy_config import DummyConfig
from tests.mocks.mock_llm import MockLLM
from tests.mocks.mock_predictor import MockPredictor


def test_predictor_predict_flow(mock_predictor):
    """Test the basic prediction flow from prompt to final prediction."""
    # Input data
    xs = np.array(["Is a ok product!"])
    prompts = ["Classify this text:"]

    # Call predict
    predictions, _ = mock_predictor.predict(prompts, xs)
    # Verify shape and content of predictions
    assert predictions.shape == (1,)
    assert predictions[0] == "neutral"

    # Verify LLM was called with correct prompts
    assert len(mock_predictor.llm.call_history) == 1
    assert mock_predictor.llm.call_history[0]["prompts"] == [
        "Classify this text:\nIs a ok product!",
    ]


def test_predictor_with_return_seq(mock_predictor):
    """Test prediction with return_seq=True."""
    # Input data
    prompts = ["Classify this text:"]
    xs = np.array(["This product is okay."])

    # Call predict with return_seq=True
    predictions, sequences = mock_predictor.predict(prompts, xs)

    # Verify predictions
    assert predictions.shape == (1,)
    assert predictions[0] == "neutral"

    # Verify sequences
    assert len(sequences) == 1
    assert isinstance(sequences, list)
    assert "This product is okay." in sequences[0]


def test_predictor_accepts_string_prompt(mock_predictor):
    preds, seqs = mock_predictor.predict("solo", ["input"], system_prompts="sys")
    assert preds.shape[0] == 1
    assert seqs[0].startswith("input\n")


def test_predictor_system_prompt_string_converted(mock_predictor):
    preds, seqs = mock_predictor.predict(["p1", "p2"], ["x1", "x2"], system_prompts="sys")
    assert len(preds) == 2
    # call_history should show system_prompts broadcasted
    assert mock_predictor.llm.call_history[-1]["system_prompts"] == ["sys", "sys"]


def test_predictor_applies_config():
    cfg = DummyConfig()
    predictor = MockPredictor(llm=MockLLM(), config=cfg)
    assert cfg.applied is True
    assert getattr(predictor, "config_applied") is True
