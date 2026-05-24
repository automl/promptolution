# Getting Started: LLM as a Judge with Promptolution

## Welcome to Promptolution! 

Discover a powerful tool for evolving and optimizing your LLM prompts. This notebook provides a friendly introduction to one of Promptolution's most advanced features: LLM as a Judge.

While the standard getting_started notebook shows how to optimize for classification tasks, this guide will focus on something different. We'll optimize prompts for a creative task where there's no single "correct" answer: *Finding an optimal argument for a statement*!

## Intro
In traditional machine learning and prompt optimization, we often rely on labeled data. For a classification task, you need an input (x) and a corresponding ground-truth label (y). The goal is to find a prompt that helps the model predict y correctly.
But what if your task is more subjective? How do you "label" things like:

- The quality of a generated argument?
- The creativity of a story?
- The helpfulness of a summary?
- The persuasiveness of an essay?

This is where LLM as a Judge comes in. Instead of relying on a pre-defined dataset of labels, we use another powerful Language Model (the "judge") to score the output of our prompts. The process looks like this:

A candidate prompt is used to generate a response (e.g., an argument).
A "judge" LLM then evaluates this response based on the task provided and assigns a score.
Promptolution's optimizer uses these scores to identify which prompts are best and evolves them to generate even better responses.

The beauty of this approach is its flexibility. While you can provide groundtruths (in case there is a correct answer) and let the LLM judge itself if both the prediction and the correct answer are equivalent - you don't need to.

*New to Promptolution? If you haven't seen our classification tutorial yet, check out `getting_started.ipynb` first! It covers the basics of prompt optimization with simpler tasks like text classification. This notebook builds on those concepts but tackles more complex, subjective tasks.*

## Installation
Install Promptolution with a single command


```python
! pip install promptolution[api]
```

## Imports


```python
import pandas as pd
from promptolution.utils import ExperimentConfig
from promptolution.helpers import run_experiment
import nest_asyncio

nest_asyncio.apply()  # Required for notebook environments
```

## Setting Up Your Experiment

### Prepare the data

For this tutorial, we're using IBM's Argument Quality Ranking dataset - a collection of crowd-sourced arguments on controversial topics like capital punishment, abortion rights, and climate change.

Unlike classification tasks where you have clear input-output pairs, here we're working with debate topics that we want to generate compelling arguments for.


```python
df = pd.read_csv("hf://datasets/ibm-research/argument_quality_ranking_30k/dev.csv").sample(300)
```


```python
print("\nSample topics:")
for topic in df["topic"].unique()[:3]:
    print(f"- {topic}")
```

    
    Sample topics:
    - We should adopt a zero-tolerance policy in schools
    - Payday loans should be banned
    - Intelligence tests bring more harm than good
    

Our task: **Given a controversial statement, generate the strongest possible argument supporting that position.**

Let's look at what we're working with:

### Creating Initial Prompts

Here are some starter prompts for generating compelling arguments. Feel free to experiment with your own!


```python
init_prompts = [
    "Create a strong argument for this position with clear reasoning and examples:",
    "Write a persuasive argument supporting this statement. Include evidence and address counterarguments:",
    "Make a compelling case for this viewpoint using logical reasoning and real examples:",
    "Argue convincingly for this position. Provide supporting points and evidence:",
    "Build a strong argument for this statement with clear structure and solid reasoning:",
    "Generate a persuasive argument supporting this position. Use facts and logical flow:",
    "Create a well-reasoned argument for this viewpoint with supporting evidence:",
    "Write a convincing argument for this position. Include examples and counter opposing views:",
    "Develop a strong case supporting this statement using clear logic and evidence:",
    "Construct a persuasive argument for this position with solid reasoning and examples:",
]
```

### Configure Your LLM

For this demonstration, we will again use the DeepInfra API, but you can easily switch to other providers like Anthropic or OpenAI by simply changing the `api_url` and `model_id`.


```python
api_key = "YOUR_API_KEY"  # Replace with your Promptolution API key
```

Here are the key parameters for LLM-as-a-Judge tasks:


```python
config = ExperimentConfig(
    optimizer="evopromptga",
    task_description="Given a statement, find the best argument supporting it.",
    x_column="topic",
    prompts=init_prompts,
    n_steps=3,
    n_subsamples=10,
    subsample_strategy="random_subsample",
    api_url="https://api.deepinfra.com/v1/openai",
    model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    api_key=api_key,
    task_type="judge",
)
```

- `task_type="judge"` - This tells Promptolution to use LLM evaluation instead of accuracy metrics
- `x_column="topic"` - We specify which column contains our input (debate topics)
- `optimizer="evopromptga"` - In the classification task we show cased CAPO, here we are using EvoPrompt, a strong evolutionary prompt optimizer.
- No y column needed - the judge will evaluate quality without ground truth labels!

## Run Your Experiment

With everything configured, you're ready to optimize your prompts! The run_experiment function will:

1. Evaluate your initial prompts by generating arguments and having the judge LLM score them
1. Use evolutionary operators (mutation, crossover) to create new prompt variations from the 1. best-performing ones
1. Test these new prompt candidates and select the fittest ones for the next generation
1. Repeat this evolutionary process for the specified number of steps, gradually improving prompt 1. quality


```python
prompts = run_experiment(df, config)
```

    🔥 Starting optimization...
    

You can expect this to take several minutes as the optimizer generates arguments, evaluates them with the judge, and evolves the prompts.


```python
prompts
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prompt</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Construct a persuasive argument supporting the given statement, relying on logical coherence and evidence-based reasoning.</td>
      <td>0.931500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Develop a strong case supporting this statement using clear logic and evidence:</td>
      <td>0.924167</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Construct a convincing case supporting the stated argument, providing evidence and responding to potential objections.</td>
      <td>0.915833</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Develop a well-reasoned argument in favor of the given statement, incorporating reliable examples and addressing potential counterpoints.</td>
      <td>0.913333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Write a persuasive argument supporting this statement. Include evidence and address counterarguments:</td>
      <td>0.907500</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Present a convincing case for this assertion, incorporating logical premises and applicable examples.</td>
      <td>0.903333</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Fortify the provided statement with a robust and well-reasoned argument, underscoring logical relationships and leveraging empirical support to build a compelling case, while also anticipating and addressing potential counterpoints.</td>
      <td>0.902500</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Construct a strong claim in support of this statement, employing a logical framework and relevant examples to make a convincing case.</td>
      <td>0.891667</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Create a well-reasoned argument for this viewpoint with supporting evidence:</td>
      <td>0.888333</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Extract the most compelling supporting argument for this statement, grounding it in logical reasoning and bolstered by relevant evidence and examples.</td>
      <td>0.697500</td>
    </tr>
  </tbody>
</table>
</div>



The best prompts aren't always the most obvious ones - let the optimizer surprise you with what works!


Happy prompt optimizing! 🚀✨ We can't wait to see what you build with Promptolution! 🤖💡


```python

```
