"""Module for task-related functions and classes."""

from promptolution.tasks.classification_tasks import ClassificationTask
from promptolution.tasks.judge_tasks import JudgeTask
from promptolution.tasks.reward_tasks import RewardTask

__all__ = [
    "ClassificationTask",
    "JudgeTask",
    "RewardTask",
]
