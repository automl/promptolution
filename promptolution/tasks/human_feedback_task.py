"""Module for human-feedback tasks driven by a Gradio web UI."""


from __future__ import annotations

import threading
from collections import defaultdict

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from promptolution.tasks.base_task import BaseTask
from promptolution.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.tasks.base_task import EvalStrategy
    from promptolution.utils.config import ExperimentConfig


logger = get_logger(__name__)

FeedbackMode = Literal["rate", "rank"]

_GRADIO_INSTALL_HINT = (
    "HumanFeedbackTask needs gradio. Install it with "
    "`poetry install --with gui` or `pip install gradio`."
)


class HumanFeedbackTask(BaseTask):
    """Task that scores LLM responses through a Gradio web UI.

    Two interaction modes are supported:

    - ``"rate"``: each (input, prediction) pair is shown one at a time and
      rated on a Likert scale between ``min_score`` and ``max_score``. The
      rating is linearly normalized to ``[0, 1]``.
    - ``"rank"``: all predictions belonging to the same input are shown
      together and assigned ranks 1..N (1 = best). The best rank gets a
      score of ``1.0``, the worst ``0.0``; single-prediction groups get
      ``1.0`` automatically.

    The Gradio app launches in the user's browser and blocks ``evaluate``
    until every item has been labeled.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        x_column: str = "x",
        y_column: Optional[str] = None,
        task_description: Optional[str] = None,
        n_subsamples: int = 30,
        eval_strategy: "EvalStrategy" = "full",
        seed: int = 42,
        mode: FeedbackMode = "rate",
        min_score: int = 1,
        max_score: int = 5,
        server_port: Optional[int] = None,
        share: bool = False,
        inbrowser: bool = True,
        config: Optional["ExperimentConfig"] = None,
    ) -> None:
        """Initialize the HumanFeedbackTask.

        Args:
            df: Input DataFrame containing the data.
            x_column: Name of the column with input texts.
            y_column: Optional column with reference outputs to display.
            task_description: Description shown to the human evaluator.
            n_subsamples: Number of subsamples to use for evaluation.
            eval_strategy: Subsampling strategy.
            seed: Random seed for reproducibility.
            mode: ``"rate"`` for Likert scoring, ``"rank"`` for relative ranking.
            min_score: Lowest Likert value (rate mode only).
            max_score: Highest Likert value (rate mode only).
            server_port: Port for the Gradio server (auto if None).
            share: Whether to expose a public Gradio share link.
            inbrowser: Whether to open a browser tab automatically.
            config: Optional config to override defaults.
        """
        if mode not in ("rate", "rank"):
            raise ValueError(f"mode must be 'rate' or 'rank', got '{mode}'.")
        if max_score <= min_score:
            raise ValueError("max_score must be strictly greater than min_score.")

        self.mode: FeedbackMode = mode
        self.min_score = min_score
        self.max_score = max_score
        self.server_port = server_port
        self.share = share
        self.inbrowser = inbrowser

        super().__init__(
            df=df,
            x_column=x_column,
            y_column=y_column,
            task_description=task_description,
            n_subsamples=n_subsamples,
            eval_strategy=eval_strategy,
            seed=seed,
            config=config,
        )
        self.task_type = "reward"

    def _launch_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "share": self.share,
            "inbrowser": self.inbrowser,
            "quiet": True,
            "prevent_thread_lock": True,
        }
        if self.server_port is not None:
            kwargs["server_port"] = self.server_port
        return kwargs

    def _evaluate(self, xs: List[str], ys: List[str], preds: List[str]) -> np.ndarray:
        if not preds:
            return np.asarray([], dtype=float)

        if self.mode == "rate":
            items = [
                {"x": x, "y": (y if self.has_y else None), "pred": pred}
                for x, y, pred in zip(xs, ys, preds)
            ]
            ratings = _run_rate_gui(
                task_description=self.task_description,
                items=items,
                min_score=self.min_score,
                max_score=self.max_score,
                launch_kwargs=self._launch_kwargs(),
            )
            denom = self.max_score - self.min_score
            return np.asarray(
                [(r - self.min_score) / denom for r in ratings], dtype=float
            )

        # mode == "rank": group predictions by input, label one group per page.
        groups: "defaultdict[str, List[int]]" = defaultdict(list)
        for i, x in enumerate(xs):
            groups[x].append(i)

        group_data = [
            {
                "x": x,
                "y": (ys[indices[0]] if self.has_y else None),
                "preds": [preds[i] for i in indices],
                "original_indices": indices,
            }
            for x, indices in groups.items()
        ]

        orderings = _run_rank_gui(
            task_description=self.task_description,
            groups=group_data,
            launch_kwargs=self._launch_kwargs(),
        )

        scores = np.zeros(len(preds), dtype=float)
        for group, order in zip(group_data, orderings):
            n = len(order)
            indices = group["original_indices"]
            if n == 1:
                scores[indices[0]] = 1.0
                continue
            for rank_position, local_idx in enumerate(order):
                scores[indices[local_idx]] = 1.0 - rank_position / (n - 1)
        return scores


def _run_rate_gui(
    task_description: Optional[str],
    items: List[Dict[str, Any]],
    min_score: int,
    max_score: int,
    launch_kwargs: Dict[str, Any],
) -> List[float]:
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_GRADIO_INSTALL_HINT) from exc

    n = len(items)
    show_gt = any(item["y"] is not None for item in items)
    results: List[float] = [float(min_score)] * n
    done = threading.Event()

    first = items[0]

    with gr.Blocks(title="Human feedback - rate responses") as demo:
        idx_state = gr.State(0)

        gr.Markdown(f"### {task_description or 'Rate each response.'}")
        progress = gr.Markdown(f"**Item 1 of {n}**")

        input_box = gr.Textbox(
            label="Input", value=first["x"], interactive=False, lines=3
        )
        gt_box = gr.Textbox(
            label="Ground truth",
            value=(first["y"] or "") if show_gt else "",
            interactive=False,
            lines=2,
            visible=show_gt,
        )
        pred_box = gr.Textbox(
            label="Model response", value=first["pred"], interactive=False, lines=8
        )
        rating = gr.Radio(
            choices=list(range(min_score, max_score + 1)),
            label=f"Rating ({min_score} = worst, {max_score} = best)",
            value=min_score,
        )
        submit = gr.Button("Next", variant="primary")

        def on_submit(idx: int, rating_val: int):
            results[idx] = float(rating_val)
            next_idx = idx + 1
            if next_idx >= n:
                done.set()
                return (
                    f"**Done — all {n} responses rated. You can close this tab.**",
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(value=min_score, interactive=False),
                    gr.update(interactive=False),
                    next_idx,
                )
            nxt = items[next_idx]
            return (
                f"**Item {next_idx + 1} of {n}**",
                nxt["x"],
                (nxt["y"] or "") if show_gt else "",
                nxt["pred"],
                gr.update(value=min_score),
                gr.update(),
                next_idx,
            )

        submit.click(
            on_submit,
            inputs=[idx_state, rating],
            outputs=[progress, input_box, gt_box, pred_box, rating, submit, idx_state],
        )

    demo.queue().launch(**launch_kwargs)
    try:
        done.wait()
    finally:
        demo.close()
    return results


def _run_rank_gui(
    task_description: Optional[str],
    groups: List[Dict[str, Any]],
    launch_kwargs: Dict[str, Any],
) -> List[List[int]]:
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_GRADIO_INSTALL_HINT) from exc

    n_groups = len(groups)
    orderings: List[List[int]] = [list(range(len(g["preds"]))) for g in groups]
    done = threading.Event()

    def _make_table(group: Dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Rank (1=best)": list(range(1, len(group["preds"]) + 1)),
                "Response": group["preds"],
            }
        )

    first = groups[0]
    show_gt = first["y"] is not None

    with gr.Blocks(title="Human feedback - rank responses") as demo:
        idx_state = gr.State(0)

        gr.Markdown(f"### {task_description or 'Rank the responses (1 = best).'}")
        progress = gr.Markdown(f"**Group 1 of {n_groups}**")
        input_md = gr.Markdown(f"**Input:**\n\n{first['x']}")
        gt_md = gr.Markdown(
            f"**Ground truth:**\n\n{first['y']}" if show_gt else "",
            visible=show_gt,
        )

        table = gr.Dataframe(
            value=_make_table(first),
            headers=["Rank (1=best)", "Response"],
            datatype=["number", "str"],
            interactive=True,
            wrap=True,
        )
        status = gr.Markdown()
        submit = gr.Button("Confirm ranking", variant="primary")

        def on_submit(idx: int, table_df: pd.DataFrame):
            n = len(groups[idx]["preds"])
            try:
                ranks = [int(r) for r in table_df["Rank (1=best)"].tolist()]
            except (ValueError, TypeError):
                return (
                    f"❌ Ranks must be integers between 1 and {n}.",
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    idx,
                )
            if sorted(ranks) != list(range(1, n + 1)):
                return (
                    f"❌ Ranks must be a permutation of 1..{n}, got {ranks}.",
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    idx,
                )

            order = [0] * n
            for original_idx, rank in enumerate(ranks):
                order[rank - 1] = original_idx
            orderings[idx] = order

            next_idx = idx + 1
            if next_idx >= n_groups:
                done.set()
                return (
                    f"**Done — all {n_groups} groups ranked. You can close this tab.**",
                    gr.update(),
                    gr.update(),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(value="✓ Saved."),
                    next_idx,
                )

            nxt = groups[next_idx]
            nxt_show_gt = nxt["y"] is not None
            return (
                f"**Group {next_idx + 1} of {n_groups}**",
                f"**Input:**\n\n{nxt['x']}",
                gr.update(
                    value=(f"**Ground truth:**\n\n{nxt['y']}" if nxt_show_gt else ""),
                    visible=nxt_show_gt,
                ),
                _make_table(nxt),
                gr.update(),
                gr.update(value=""),
                next_idx,
            )

        submit.click(
            on_submit,
            inputs=[idx_state, table],
            outputs=[progress, input_md, gt_md, table, submit, status, idx_state],
        )

    demo.queue().launch(**launch_kwargs)
    try:
        done.wait()
    finally:
        demo.close()
    return orderings
