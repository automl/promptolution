"""Tutorial: optimize prompts using human feedback collected through a Tkinter GUI.

For each evaluation step the GUI pops up and asks you to either rate or rank
the model's responses. The collected scores feed straight back into the
optimizer just like any other task.
"""


import argparse
from logging import Logger

import pandas as pd

from promptolution.llms import APILLM
from promptolution.optimizers import EvoPromptGA
from promptolution.predictors import MarkerBasedPredictor
from promptolution.tasks import HumanFeedbackTask
from promptolution.utils import LoggerCallback

logger = Logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="https://openwebui.uni-freiburg.de/api")
parser.add_argument("--model", default="qwen-30b-a3b-llmlb")
parser.add_argument("--token", default=None, help="API token for the OpenWebUI gateway.")
parser.add_argument("--mode", choices=["rate", "rank"], default="rank")
parser.add_argument("--n-steps", type=int, default=2)
args = parser.parse_args()

# A tiny hand-curated dataset of prompts the LLM should respond to.
# Keep it small — every evaluation triggers a human-in-the-loop GUI window.
df = pd.DataFrame(
    {
        "input": [
            "Explain what overfitting is to a high-school student.",
            "Suggest three creative names for a coffee shop near a university.",
            "Write a haiku about debugging code at 3am.",
            "Summarize the plot of 'The Lord of the Rings' in two sentences.",
        ]
    }
)

task = HumanFeedbackTask(
    df=df,
    x_column="input",
    task_description=(
        "You will see a model response to a short writing/explanation task. "
        "Rate how clear, correct, and engaging the response is."
    ),
    mode=args.mode,           # "rate" -> per-response Likert; "rank" -> best-to-worst within an input
    min_score=1,
    max_score=5,
    n_subsamples=len(df),     # use the whole tiny dataset every round
    eval_strategy="full",
)

initial_prompts = [
    "Answer the user's request clearly and concisely. Put your final reply between <final_answer> and </final_answer>.",
    "You are a thoughtful assistant. Respond helpfully and vividly. Wrap your final reply in <final_answer> </final_answer>.",
    "Reply in a friendly, conversational tone. Keep it short. Place your answer inside <final_answer> </final_answer> tags.",
    "Provide an accurate, well-structured answer in plain language, inside <final_answer> </final_answer> markers.",
]

llm = APILLM(api_url=args.base_url, model_id=args.model, api_key=args.token)

# Free-form text task: extract whatever the model puts between the markers
# (no fixed class list) so the human sees the LLM's chosen answer.
predictor = MarkerBasedPredictor(llm)

callbacks = [LoggerCallback(logger)]

optimizer = EvoPromptGA(
    task=task,
    predictor=predictor,
    meta_llm=llm,
    initial_prompts=initial_prompts,
    callbacks=callbacks,
)

best_prompts = optimizer.optimize(n_steps=args.n_steps)

print("Top prompts after human-in-the-loop optimization:")
for rank, prompt in enumerate(best_prompts, start=1):
    print(f"  #{rank}: {prompt}")
