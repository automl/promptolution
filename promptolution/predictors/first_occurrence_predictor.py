"""Module for the FirstOccurrencePredictor."""

from typing import TYPE_CHECKING, List, Optional

from promptolution.predictors.base_predictor import BasePredictor

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.llms.base_llm import BaseLLM
    from promptolution.utils.config import ExperimentConfig


class FirstOccurrencePredictor(BasePredictor):
    """A predictor class for classification tasks using language models.

    This class takes a language model and a list of classes, and provides a method
    to predict classes for given prompts and input data. The class labels are extracted
    by matching the words in the prediction with the list of valid class labels.
    The first occurrence of a valid class label in the prediction is used as the predicted class.
    If no valid class label is found, the first class label in the list is used as the default prediction.

    Attributes:
        llm: The language model used for generating predictions.
        classes (List[str]): The list of valid class labels.
        config (ExperimentConfig, optional): Configuration for the classifier, overriding defaults.

    Inherits from:
        BasePredictor: The base class for predictors in the promptolution library.
    """

    def __init__(self, llm: "BaseLLM", classes: List[str], config: Optional["ExperimentConfig"] = None) -> None:
        """Initialize the FirstOccurrencePredictor.

        Args:
            llm: The language model to use for predictions.
            classes (List[str]): The list of valid class labels.
            config (ExperimentConfig, optional): Configuration for the classifier, overriding defaults.
        """
        assert all([c.islower() for c in classes]), "Class labels should be lowercase."
        self.classes = classes

        self.extraction_description = (
            f"The task is to classify the texts into one of those classes: {', '.join(classes)}."
            "The first occurrence of a valid class label in the prediction is used as the predicted class."
        )

        super().__init__(llm, config)

    def _extract_preds(self, preds: List[str]) -> List[str]:
        """Extract class labels from the predictions, based on the list of valid class labels.

        Args:
            preds: The raw predictions from the language model.
        """
        result = []
        for pred in preds:
            predicted_class = self.classes[0]  # use first class as default pred
            for word in pred.split():
                word = "".join([c for c in word if c.isalnum()]).lower()
                if word in self.classes:
                    predicted_class = word
                    break

            result.append(predicted_class)

        return result
