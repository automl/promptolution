import pytest

from tests.mocks.dummy_config import DummyConfig
from tests.mocks.mock_predictor import MockPredictor
from tests.mocks.mock_task import MockTask

from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.utils.callbacks import BaseCallback


class SimpleOptimizer(BaseOptimizer):
    def __init__(self, predictor, task, **kwargs):
        super().__init__(predictor=predictor, task=task, initial_prompts=["p1", "p2"], **kwargs)
        self.prepared = False
        self.steps = 0

    def _pre_optimization_loop(self):
        self.prepared = True

    def _step(self):
        self.steps += 1
        return self.prompts


class FailingOptimizer(SimpleOptimizer):
    def _step(self):
        raise RuntimeError("boom")

    def _on_train_end(self):
        self.cleaned = True
        return None


class Stopper(BaseCallback):
    def on_step_end(self, optimizer):
        # stop after first step to exercise callback stop path
        return False

    def on_train_end(self, optimizer):
        optimizer.stopped = True
        return True


@pytest.fixture
def predictor():
    return MockPredictor()


@pytest.fixture
def task():
    return MockTask()


def test_base_optimizer_runs_and_calls_callbacks(predictor: MockPredictor, task: MockTask):
    opt = SimpleOptimizer(predictor=predictor, task=task)
    opt.callbacks = [Stopper()]
    opt.optimize(3)

    assert opt.prepared is True
    assert opt.steps == 1
    assert getattr(opt, "stopped", False) is True


def test_base_optimizer_stops_on_exception(predictor: MockPredictor, task: MockTask):
    opt = FailingOptimizer(predictor=predictor, task=task)
    opt.optimize(2)

    assert opt.prepared is True
    assert getattr(opt, "cleaned", False) is True


def test_base_optimizer_no_callbacks_continues(predictor: MockPredictor, task: MockTask):
    opt = SimpleOptimizer(predictor=predictor, task=task)
    opt.optimize(2)
    assert opt.steps == 2


def test_base_optimizer_config_validate_and_template(predictor: MockPredictor, task: MockTask):
    cfg = DummyConfig(task_description="override desc")
    opt = SimpleOptimizer(predictor=predictor, task=task, config=cfg)
    opt.optimize(1)
    assert cfg.validated is True

    templ = opt._initialize_meta_template("Hi <task_desc>")
    assert "override desc" in templ
    assert getattr(opt, "config_applied", True)
