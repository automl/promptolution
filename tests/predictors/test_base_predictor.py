import numpy as np


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
