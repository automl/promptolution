"""Helper functions for the usage of the libary."""

from typing import TYPE_CHECKING, Callable, List, Literal, Optional, Union, cast

from promptolution.tasks.judge_tasks import JudgeTask
from promptolution.tasks.reward_tasks import RewardTask
from promptolution.utils.prompt import Prompt
from promptolution.utils.prompt_creation import create_prompts_from_task_description

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.exemplar_selectors.base_exemplar_selector import BaseExemplarSelector
    from promptolution.llms.base_llm import BaseLLM
    from promptolution.optimizers.base_optimizer import BaseOptimizer
    from promptolution.predictors.base_predictor import BasePredictor
    from promptolution.tasks.base_task import BaseTask
    from promptolution.utils.config import ExperimentConfig
    from promptolution.tasks.base_task import TaskType
    from promptolution.optimizers.base_optimizer import OptimizerType
    from promptolution.predictors.base_predictor import PredictorType

import pandas as pd

from promptolution.exemplar_selectors.random_search_selector import RandomSearchSelector
from promptolution.exemplar_selectors.random_selector import RandomSelector
from promptolution.llms.api_llm import APILLM
from promptolution.llms.local_llm import LocalLLM
from promptolution.llms.vllm import VLLM
from promptolution.optimizers.capo import CAPO
from promptolution.optimizers.capoeira import Capoeira
from promptolution.optimizers.evoprompt_de import EvoPromptDE
from promptolution.optimizers.evoprompt_ga import EvoPromptGA
from promptolution.optimizers.opro import OPRO
from promptolution.predictors.first_occurrence_predictor import FirstOccurrencePredictor
from promptolution.predictors.maker_based_predictor import MarkerBasedPredictor
from promptolution.tasks.classification_tasks import ClassificationTask
from promptolution.utils.logging import get_logger

logger = get_logger(__name__)


def run_experiment(df: pd.DataFrame, config: "ExperimentConfig") -> pd.DataFrame:
    """Run a full experiment based on the provided configuration.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        config (Config): Configuration object for the experiment.

    Returns:
        pd.DataFrame: A DataFrame containing the prompts and their scores.
    """
    # train test split
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    prompts = run_optimization(train_df, config)
    prompts_str = [p.construct_prompt() for p in prompts]
    df_prompt_scores = run_evaluation(test_df, config, prompts_str)

    return df_prompt_scores


def run_optimization(df: pd.DataFrame, config: "ExperimentConfig") -> List[Prompt]:
    """Run the optimization phase of the experiment.

    Configures all LLMs (downstream, meta, and judge) to use
    the same instance, that is defined in `config.llm`.

    Args:
        config (Config): Configuration object for the experiment.

    Returns:
        List[Prompt]: The optimized list of prompts.
    """
    llm = get_llm(config=config)
    predictor = get_predictor(llm, config=config)

    if getattr(config, "prompts") is None:
        initial_prompts = create_prompts_from_task_description(
            task_description=config.task_description,
            llm=llm,
        )
        config.prompts = [Prompt(p) for p in initial_prompts]

    if config.optimizer in {"capo", "capoeira"} and (config.eval_strategy is None or "block" not in config.eval_strategy):
        logger.warning("📌 CAPO-style optimizers require block evaluation strategy. Setting it to 'sequential_block'.")
        config.eval_strategy = "sequential_block"

    task = get_task(df, config, judge_llm=llm)
    optimizer = get_optimizer(
        predictor=predictor,
        meta_llm=llm,
        task=task,
        config=config,
    )
    logger.warning("🔥 Starting optimization...")
    prompts = optimizer.optimize(n_steps=config.n_steps)

    if hasattr(config, "posthoc_exemplar_selection") and config.posthoc_exemplar_selection:
        selector = get_exemplar_selector(config.exemplar_selector, task, predictor)
        prompts = [selector.select_exemplars(p, n_examples=config.n_exemplars) for p in prompts]
    return prompts


def run_evaluation(
    df: pd.DataFrame, config: "ExperimentConfig", prompts: Union[List[Prompt], List[str]]
) -> pd.DataFrame:
    """Run the evaluation phase of the experiment.

    Configures all LLMs (downstream, meta, and judge) to use
    the same instance, that is defined in `config.llm`.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        config (Config): Configuration object for the experiment.
        prompts (List[str]): List of prompts to evaluate.

    Returns:
        pd.DataFrame: A DataFrame containing the prompts and their scores.
    """
    llm = get_llm(config=config)
    task = get_task(df, config, judge_llm=llm)
    predictor = get_predictor(llm, config=config)
    logger.warning("📊 Starting evaluation...")
    if isinstance(prompts[0], str):
        str_prompts = cast(List[str], prompts)
        prompts = [Prompt(p) for p in str_prompts]
    else:
        str_prompts = [p.construct_prompt() for p in cast(List[Prompt], prompts)]
    scores = task.evaluate(prompts, predictor, eval_strategy="full")
    df = pd.DataFrame(dict(prompt=str_prompts, score=scores))
    df = df.sort_values("score", ascending=False, ignore_index=True)

    return df


def get_llm(model_id: Optional[str] = None, config: Optional["ExperimentConfig"] = None) -> "BaseLLM":
    """Factory function to create and return a language model instance based on the provided model_id.

    This function supports three types of language models:
    1. LocalLLM: For running models locally.
    2. VLLM: For running models using the vLLM library.
    3. APILLM: For API-based models (default if not matching other types).

    Args:
        model_id (str): Identifier for the model to use. Special cases:
                        - "local-{model_name}" for LocalLLM
                        - "vllm-{model_name}" for VLLM
                        - Any other string for APILLM
        config (ExperimentConfig, optional): "ExperimentConfig" overwriting defaults.

    Returns:
        An instance of LocalLLM, or APILLM based on the model_id.
    """
    final_model_id = model_id or (config.model_id if config else None)
    if not final_model_id:
        raise ValueError("model_id must be provided either directly or through config.")

    if "local" in final_model_id:
        model_name = "-".join(final_model_id.split("-")[1:])
        return LocalLLM(model_name, config=config)
    if "vllm" in final_model_id:
        model_name = "-".join(final_model_id.split("-")[1:])
        return VLLM(model_name, config=config)

    return APILLM(model_id=final_model_id, config=config)


def get_task(
    df: pd.DataFrame,
    config: "ExperimentConfig",
    task_type: Optional["TaskType"] = None,
    judge_llm: Optional["BaseLLM"] = None,
    reward_function: Optional[Callable] = None,
) -> "BaseTask":
    """Get the task based on the provided DataFrame and configuration.

    So far only ClassificationTask is supported.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        config (ExperimentConfig): Configuration for the experiment.

    Returns:
        BaseTask: An instance of a task class based on the provided DataFrame and configuration.
    """
    final_task_type = task_type or (config.task_type if config else None)

    if final_task_type == "reward":
        if reward_function is None:
            reward_function = config.reward_function if config else None
        assert reward_function is not None, "Reward function must be provided for reward tasks."
        return RewardTask(
            df=df,
            reward_function=reward_function,
            config=config,
        )
    elif final_task_type == "judge":
        assert judge_llm is not None, "Judge LLM must be provided for judge tasks."
        return JudgeTask(df, judge_llm=judge_llm, config=config)

    return ClassificationTask(df, config=config)


def get_optimizer(
    predictor: "BasePredictor",
    meta_llm: "BaseLLM",
    task: "BaseTask",
    optimizer: Optional["OptimizerType"] = None,
    task_description: Optional[str] = None,
    config: Optional["ExperimentConfig"] = None,
) -> "BaseOptimizer":
    """Creates and returns an optimizer instance based on provided parameters.

    Args:
        predictor: The predictor used for prompt evaluation
        meta_llm: The language model used for generating meta-prompts
        task: The task object used for evaluating prompts
        optimizer: String identifying which optimizer to use
        meta_prompt: Meta prompt text for the optimizer
        task_description: Description of the task for the optimizer
        config: Configuration object with default parameters

    Returns:
        An optimizer instance

    Raises:
        ValueError: If an unknown optimizer type is specified
    """
    final_optimizer = optimizer or (config.optimizer if config else None)
    if config is None:
        config = ExperimentConfig()
    if task_description is not None:
        config.task_description = task_description

    if final_optimizer == "capo":
        return CAPO(
            predictor=predictor,
            meta_llm=meta_llm,
            task=task,
            config=config,
        )

    if final_optimizer == "capoeira":
        return Capoeira(
            predictor=predictor,
            meta_llm=meta_llm,
            task=task,
            config=config,
        )

    if final_optimizer == "evopromptde":
        return EvoPromptDE(predictor=predictor, meta_llm=meta_llm, task=task, config=config)

    if final_optimizer == "evopromptga":
        return EvoPromptGA(predictor=predictor, meta_llm=meta_llm, task=task, config=config)

    if final_optimizer == "opro":
        return OPRO(predictor=predictor, meta_llm=meta_llm, task=task, config=config)

    raise ValueError(f"Unknown optimizer: {final_optimizer}")


def get_exemplar_selector(
    name: Literal["random", "random_search"], task: "BaseTask", predictor: "BasePredictor"
) -> "BaseExemplarSelector":
    """Factory function to get an exemplar selector based on the given name.

    Args:
        name (str): The name of the exemplar selector to instantiate.
        task (BaseTask): The task object to be passed to the selector.
        predictor (BasePredictor): The predictor object to be passed to the selector.

    Returns:
        BaseExemplarSelector: An instance of the requested exemplar selector.

    Raises:
        ValueError: If the requested selector name is not found.
    """
    if name == "random_search":
        return RandomSearchSelector(task, predictor)
    elif name == "random":
        return RandomSelector(task, predictor)
    else:
        raise ValueError(f"Unknown exemplar selector: {name}")


def get_predictor(downstream_llm=None, type: "PredictorType" = "marker", *args, **kwargs) -> "BasePredictor":
    """Factory function to create and return a predictor instance.

    This function supports three types of predictors:
    1. FirstOccurrencePredictor: A predictor that classifies based on first occurrence of the label.
    2. MarkerBasedPredictor: A predictor that classifies based on a marker.

    Args:
        downstream_llm: The language model to use for prediction.
        type (Literal["first_occurrence", "marker"]): The type of predictor to create:
                    - "first_occurrence" for FirstOccurrencePredictor
                    - "marker" (default) for MarkerBasedPredictor
        *args: Variable length argument list passed to the predictor constructor.
        **kwargs: Arbitrary keyword arguments passed to the predictor constructor.

    Returns:
        An instance of FirstOccurrencePredictor or MarkerBasedPredictor.
    """
    if type == "first_occurrence":
        return FirstOccurrencePredictor(downstream_llm, *args, **kwargs)
    elif type == "marker":
        return MarkerBasedPredictor(downstream_llm, *args, **kwargs)
    else:
        raise ValueError(f"Invalid predictor type: '{type}'")
