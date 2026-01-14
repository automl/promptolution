# Getting Started: Reward Tasks with Promptolution

Welcome to the world of **reward-based prompt optimization**! If you've explored our classification tutorial (`getting_started.ipynb`) or our LLM-as-a-Judge notebook (`llm_judge_getting_started.ipynb`), you've seen how to optimize prompts for predicting labels or generating content that gets rated by AI judges.

But what if you want to optimize for something completely different? What if you want to optimize for:
* **Objective, measurable outcomes** rather than subjective quality?
* **System compatibility** - does the output actually work with your software?
* **Concrete business metrics** that you can define and measure automatically?

This is where **Reward Tasks** shine. Instead of relying on pre-labeled data or AI judges, you define your own reward function - a simple Python function that takes the model's output and returns a score. The optimizer then evolves prompts that maximize this reward.

**The beauty of reward tasks**: You can optimize for literally anything you can measure! Valid JSON parsing, code execution success, mathematical correctness, format compliance, API compatibility - if you can write a function to evaluate it, you can optimize for it.

> **New to Promptolution?** If you haven't seen our other tutorials yet, check out `getting_started.ipynb` (classification) and `llm_judge_getting_started.ipynb` (LLM evaluation) first! This notebook builds on those concepts but tackles objective, measurable outcomes.

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

    c:\Users\tzehl\anaconda3\envs\d\Lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    

## Setting Up Your Experiment

### Prepare the data

For this tutorial, we're tackling a real-world challenge: summarizing text and outputting valid JSON. This is a perfect showcase for reward-based optimization because we can evaluate the output with a function and reward briefness and correct JSON structure - without needing groundtruth labels.
We're using the CNN/DailyMail dataset, which contains news articles.


```python
df = pd.read_parquet("hf://datasets/abisee/cnn_dailymail/3.0.0").sample(300)
```

Key difference from other tasks: Notice we're not using labeled "correct" JSON outputs or asking an AI to judge quality. Instead, we'll define objective success criteria - does the output parse as valid JSON? Does it contain the required fields? Is the summary concise enough for our database?

Let's explore the task:


```python
print("Dataset columns:", df.columns.tolist())
print(f"\nDataset size: {len(df)} examples")
print("\nSample Article:")
print(df["article"].iloc[0][:170] + "...")
```

    Dataset columns: ['article', 'highlights', 'id']
    
    Dataset size: 300 examples
    
    Sample Article:
    Investors looking to make an easy buck out of the housing market could be running out of time. Australia's financial regulators are in talks to tighten the process for in...
    

### Creating Inital Prompts

Here are some starter prompts for JSON extraction. Feel free to experiment with your own approaches!


```python
init_prompts = [
    """Analyze the provided news article and return a JSON response with the following three fields:
- "summary": A concise summary of the article's main points (maximum 200 characters)
- "category": The article's topic classification (options: "sports", "politics", "technology", or "other")
- "author": The article author's name (use "unknown" if not provided)
Format the response as valid JSON with these exact keys.
The final json needs to start with the <final_answer> tag.
"""
]
```

### Configure Your LLM

Promptolution offers three flexible ways to access language models:

1. Local LLMs (using the Transformers library)
1. vLLM backend (for efficient serving of large language models)
1. API-based LLMs (compatible with any provider following the OpenAI standard)

For this demonstration, we'll use the DeepInfra API, but you can easily switch to other providers like Anthropic or OpenAI by simply changing the base_url and llm string in the configuration.


```python
api_key = "YOUR_API_KEY"  # Replace with your Promptolution API key
```

Here's an explanation of each configuration parameter in the ExperimentConfig:
- `optimizer`: The algorithm used for prompt optimization. Currently we support "capo", "evopromptga", "evopromptde", and "opro". For this example, we use "capo" as it is capable of leveraging few-shot examples.
- `task_description`: A string describing the task you're optimizing prompts for. This is used to provide the meta-llm with context about your task.
- `prompts`: A list of initial prompt strings that will be used as the starting point for optimization.
- `n_steps`: The number of optimization steps to run. Higher values allow more exploration and refinement but require more API calls and computational resources.
- `api_url`: The API endpoint URL used to access the language model. This example uses DeepInfra's API which follows the OpenAI standard.
- `llm`: The LLM to use for the experiment, as both downstream and meta LLM.
- `token`: Your API authentication token required to access the language model service.

### Define Your Reward Function

This is where the magic happens! Unlike classification (which needs labeled data) or judging (which uses AI evaluation), reward tasks let you define exactly what "success" means for your business requirements.

We will reward by 0.3 the LLM for first of all creating a json that is parsable by `json.loads`.
There is an additional reward of 0.2 if the dictionary contains the key "summary" and 0.1 each for containing "category" and "author".
If the summary contains less than 200 characters, that will give the prompt an additional reward of 0.2.
We give a reward of 0.1 if the categories are correctly assigned.


```python
import json


def reward_function(prediction: str) -> float:
    reward = 0.0
    try:
        information = json.loads(prediction)
        reward += 0.3  # valid json

        if "summary" in information.keys():
            reward += 0.2  # contains summary
        if "category" in information.keys():
            reward += 0.1  # contains category
        if "author" in information.keys():
            reward += 0.1  # contains author

        if len(information.get("summary", "")) < 200:
            reward += 0.2  # summary is < 200 characters

        if information.get("category") in ["sports", "politics", "technology", "other"]:
            reward += 0.1  # category is valid
    except Exception:
        reward = 0.0

    return reward
```

This reward function captures actual business requirements - the output must be valid JSON that our systems can process, contain all required fields, respect character limits to save time for the user, and use only allowed category values.


```python
task_description = (
    "The task is to summarize a news article into a json format, that contains 'summary', 'category', and 'author'. "
    "The summary should be less than 200 characters, and the category should be one of 'sports', 'politics', 'technology', or 'other'. "
    "The final json needs to start with the <final_answer> tag."
)
```


```python
config = ExperimentConfig(
    optimizer="opro",
    task_description=task_description,
    prompts=init_prompts,
    x_column="article",
    n_steps=8,
    num_instructions_per_step=5,
    api_url="https://api.deepinfra.com/v1/openai",
    model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    api_key=api_key,
    n_subsamples=15,
    task_type="reward",
    reward_function=reward_function,
)
```

**Difference compared to Classification and LLM-As-a-Judge**:
- `task_type="reward"` - Uses your custom reward function instead of accuracy or AI judgment
- `reward_function=reward_function` - Your objective success criteria
- `optimizer="opro"` - We already used EvoPrompt and CAPO in the other tutorials - here we will use OPRO. Its main benefit: it requires only a single initial prompt.
- No need for labeled "correct" outputs - the reward function defines success
- Completely customizable - change the reward function to optimize for anything!

## Run Your Experiment

With everything configured, you're ready to optimize your prompts! The `run_experiment` function will run the optimization and evaluate on a holdout set. You can expect this cell to take a few minutes to run.


```python
prompts = run_experiment(df, config)
```

    🔥 Starting optimization...
    📊 Starting evaluation...
    


```python
prompts.iloc[[0, 1, 2, -2, -1]]
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
      <td>Summarize the news article into a JSON format with the following structure: {“summary”: &lt;summary&gt;, “category”: &lt;category&gt;, “author”: &lt;author&gt;}.\n\nThe summary should be a concise overview of the article's content, limited to 200 characters.\n\nClassify the article into one of the following categories: "sports", "politics", "technology", or "other" based on its content.\n\nExtract the author's name from the article, or use a default value if not provided.\n\nStart the JSON response with the tag “&lt;final_answer&gt;” and end it with “&lt;/final_answer&gt;”.</td>
      <td>0.848333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Analyze the provided news article and return a JSON response with the following three fields:\n- "summary": A concise summary of the article's main points (maximum 200 characters)\n\n- "category": The article's topic classification (options: "sports", "politics", "technology", or "other")\n\n- "author": The article author's name (use "unknown" if not provided)\n\nFormat the response as valid JSON with these exact keys.\n\nThe final json needs to start with the &lt;final_answer&gt; tag.\n</td>
      <td>0.811667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Analyze the provided news article and generate a JSON response with the following three fields:\n\n* "summary": A concise and objective summary of the article's main points, limited to 150 characters, focusing on the most critical information and highlighting key points.\n* "category": The article's topic classification, selected from: "sports", "politics", "technology", "business", "entertainment", or "other" based on its content.\n* "author": The article author's name, using "unknown" if not provided.\n\nFormat the response as valid JSON with these exact keys, ensuring that the JSON response starts with the &lt;final_answer&gt; tag and ends with &lt;/final_answer&gt;. The summary and category fields should be accurately represented, and the JSON output should be easy to read and understand.\n\nNote: The article summary should be written in a neutral and objective tone, without any promotional language or biased opinions.\n\nScore: 99</td>
      <td>0.805000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Analyze the provided news article and generate a JSON response with the following three fields:\n- "summary": A concise summary of the article's main points, limited to 250 characters, focusing on identifying the most critical information and presenting it in a clear and coherent manner.\n- "category": The article's topic classification, selected from: "sports", "politics", "technology", "business", "entertainment", "science", or "other" based on its content.\n- "author": The article author's name, using "unknown" if not provided.\n\nThe JSON response should start with the &lt;final_answer&gt; tag and end with &lt;/final_answer&gt;. Ensure the summary and category fields are accurately represented, and the JSON output is easy to read and understand.\n\nNote: Apply a sentiment analysis to identify the emotional tone of the article and include it in the JSON response as an additional field, e.g., "sentiment": "positive", "neutral", or "negative".</td>
      <td>0.711667</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Analyze the provided news article and generate a JSON response with the following three fields:\n\n* "summary": A concise summary of the article's main points, limited to 200 characters, focusing on identifying the most critical information and presenting it in a clear and coherent manner.\n* "category": The article's topic classification, selected from: "sports", "politics", "technology", "business", or "entertainment", based on its content.\n* "author": The article author's name, using a default value if not provided.\n\nFormat the response as valid JSON with these exact keys. Ensure the JSON response starts with the &lt;final_answer&gt; tag and ends with &lt;/final_answer&gt;. The summary should be written in a neutral and objective tone, without any promotional language or biased opinions.\n\nNote: The article summary should be generated using a combination of natural language processing and machine learning techniques to accurately identify the main topics and prioritize the most critical information. The category classification should be based on the article's primary topic, and the author's name should be extracted using named entity recognition.</td>
      <td>0.701667</td>
    </tr>
  </tbody>
</table>
</div>



You might think 'just ask for JSON' would work fine, but optimization reveals that specific instructions about field names, value constraints, and output formatting can improve validity rates from ~70% to over 84% - another reminder that systematic optimization beats manual prompt engineering!

Happy prompt optimizing! 🚀✨ We can't wait to see what you build with Promptolution! 🤖💡


```python

```
