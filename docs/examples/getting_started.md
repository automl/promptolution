# Getting Started with Promptolution

## Welcome to Promptolution! 

Discover a powerful tool for evolving and optimizing your LLM prompts. This notebook provides a friendly introduction to Promptolution's core functionality.

We're excited to have you try Promptolution - let's get started!

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

Below, we're using a subsample of the subjectivity dataset from Hugging Face as an example. When using your own dataset, simply ensure you name the input column "x" and the target column "y", and provide a brief description of your task, that will parsed to the meta-llm during optimization.


```python
df = pd.read_csv("hf://datasets/tasksource/subjectivity/train.csv").sample(500)
df = df.rename(columns={"Sentence": "x", "Label": "y"})
df = df.replace({"OBJ": "objective", "SUBJ": "subjective"})

task_description = (
    "The dataset contains sentences labeled as either subjective or objective. "
    "The task is to classify each sentence as either subjective or objective. "
    "The class mentioned in between the answer tags <final_answer></final_answer> will be used as the prediction."
)
```

### Creating Inital Prompts

We've defined some starter prompts below, but feel free to experiment! You might also want to explore create_prompts_from_samples to automatically generate initial prompts based on your data.


```python
init_prompts = [
    'Classify the given text as either an objective or subjective statement based on the tone and language used: e.g. the tone and language used should indicate whether the statement is a neutral, factual summary (objective) or an expression of opinion or emotional tone (subjective). Include the output classes "objective" or "subjective" in the prompt.',
    "What kind of statement is the following text: [Insert text here]? Is it <objective_statement> or <subjective_statement>?",
    'Identify whether a sentence is objective or subjective by analyzing the tone, language, and underlying perspective. Consider the emotion, opinion, and bias present in the sentence. Are the authors presenting objective facts or expressing a personal point of view? The output will be either "objective" (output class: objective) or "subjective" (output class: subjective).',
    "Classify the following sentences as either objective or subjective, indicating the name of the output classes: [input sentence]. Output classes: objective, subjective",
    '_query a text about legal or corporate-related issues, and predict whether the tone is objective or subjective, outputting the corresponding class "objective" for non-subjective language or "subjective" for subjective language_',
    'Classify a statement as either "subjective" or "objective" based on whether it reflects a personal opinion or a verifiable fact. The output classes to include are "objective" and "subjective".',
    "Classify the text as objective or subjective based on its tone and language.",
    "Classify the text as objective or subjective based on the presence of opinions or facts. Output classes: objective, subjective.",
    "Classify the given text as objective or subjective based on its tone, focusing on its intention, purpose, and level of personal opinion or emotional appeal, with outputs including classes such as objective or subjective.",
    "Categorize the text as either objective or subjective, considering whether it presents neutral information or expresses a personal opinion/bias.\n\nObjective: The text has a neutral tone and presents factual information about the actions of Democrats in Congress and the union's negotiations.\n\nSubjective: The text has a evaluative tone and expresses a positive/negative opinion/evaluation about the past performance of the country.",
    'Given a sentence, classify it as either "objective" or "subjective" based on its tone and language, considering the presence of third-person pronouns, neutral language, and opinions. Classify the output as "objective" if the tone is neutral and detached, focusing on facts and data, or as "subjective" if the tone is evaluative, emotive, or biased.',
    'Identify whether the given sentence is subjective or objective, then correspondingly output "objective" or "subjective" in the form of "<output class>, (e.g. "objective"), without quotes. Please note that the subjective orientation typically describes a sentence where the writer expresses their own opinion or attitude, whereas an objective sentence presents facts or information without personal involvement or bias. <output classes: subjective, objective>',
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


```python
config = ExperimentConfig(
    optimizer="capo",
    task_description=task_description,
    prompts=init_prompts,
    n_steps=10,
    api_url="https://api.deepinfra.com/v1/openai",
    model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    api_key=api_key,
    n_subsamples=30,
)
```

## Run Your Experiment

With everything configured, you're ready to optimize your prompts! The `run_experiment` function will run the optimization and evaluate on a holdout set. You can expect this cell to take a few minutes to run.


```python
prompts = run_experiment(df, config)
```

    📌 CAPO-style optimizers require block evaluation strategy. Setting it to 'sequential_block'.
    ⚠️ The LLM does not have a tokenizer. Using simple token count.
    🔥 Starting optimization...
    📊 Starting evaluation...
    

As you can see, most optimized prompts are semantically very similar, however they often differ heavily in performance. This is exactly what we observed in our experiments across various LLMs and datasets. Running prompt optimization is an easy way to gain significant performance improvements on your task for free!

If you run into any issues while using Promptolution, please feel free to contact us. We're also happy to receive support through pull requests and other contributions to the project.


Happy prompt optimizing! 🚀✨ We can't wait to see what you build with Promptolution! 🤖💡


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
      <td>Classify the text as objective or subjective based on the presence of opinions or facts. Output classes: objective, subjective.\n\nInput:\nThe proposed agreement includes the best wage increases for rail workers in over forty years.\n\nOutput:\nobjective\n\nInput:\nThe principal reason, from the point of view of government, is that a universal income tax would be a powerful restraint upon the expansion of government.\n\nOutput:\n&lt;final_answer&gt;objective&lt;/final_answer&gt;\n\nInput:</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Task: Linguistic Analysis for Sentence Classification\n\nClassify each sentence as either objective or subjective by applying linguistic insights to identify its tone, emotion, and degree of neutrality. Examine the sentences' language features, sentiment, and presence of verifiable facts or personal opinions. Determine whether each sentence presents impartial data or conveys the author's emotions, beliefs, or biases. Treat each sentence as a distinct entity, analyzing its contours, nuances, and purpose. Consider the distinction between factual reports like news articles and opinion-based writings like blog posts. Make a nuanced classification by scrutinizing the sentence's impact, intention, and emotional resonance.\n\nYour response should be comprised of two parts: the classification and the rationale. Enclose the first-mentioned class within the markers &lt;final_answer&gt; and &lt;/final_answer&gt;. For instance, if the classification is 'objective', the output should be &lt;final_answer&gt;objective&lt;/final_answer&gt;. Focus on the sentence's language, tone, and emotional appeal to make an informed decision about its categorization, prioritizing the sentence's intention and purpose.\n\nInput:\nThe last may go very deep.\n\nOutput:\n&lt;final_answer&gt;objective&lt;/final_answer&gt;\n\nInput:\n“This latest rule will open our borders even more, and the Court seems to relish making arbitrary decisions without thinking about consequences.\n\nOutput:\n&lt;final_answer&gt;objective&lt;/final_answer&gt;\n\nInput:</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Classify each sentence as either objective or subjective by unpacking its linguistic nuances and emotional undertones. Analyze the sentence's language features, sentiment, and presence of verifiable facts or personal opinions to determine whether it presents impartial data or conveys the author's emotions, beliefs, or biases. Treat each sentence as a standalone entity, examining its contours, subtleties, and intended purpose. Consider the distinction between factual reporting, like news articles, and opinion-based writings, like blog posts. Make a refined classification by scrutinizing the sentence's impact, intention, and emotional resonance, prioritizing the sentence's intention and purpose. Your response should consist of two parts: the classification and the rationale. Enclose the primary classification within the markers &lt;final_answer&gt; and &lt;/final_answer&gt;. Focus on the sentence's language, tone, and emotional appeal to make an informed decision about its categorization. Classify each sentence as either objective or subjective by examining its linguistic tone, underlying intent, and purpose. Determine whether the text presents a neutral, factual account or expresses a personal opinion or emotional bias. Evaluate whether the text provides a neutral, factual report or reveals an evaluative tone, offering a positive or negative appraisal. Outputs will include classifications like objective or subjective, with the initial response serving as the prediction.\n\nInput:\nOver several decades, Prime Central London – or PCL – had become a repository for cash from wealthy foreigners, whether they actually wanted to live there or not.\n\nOutput:\n&lt;final_answer&gt;objective&lt;/final_answer&gt;\n\nInput:</td>
      <td>0.71</td>
    </tr>
    <tr>
      <th>3</th>
      <td>&lt;promptгалтер/&gt;\n\nClassify each sentence as either objective or subjective by examining its linguistic tone, underlying intent, and purpose. Consider whether the text presents a neutral, factual account or expresses a personal opinion or emotional bias. Evaluate whether the text is neutral and provides mere reportage, such as a factual report on congressional Democrats' actions and labor union negotiations, or if it reveals an evaluative tone, offering a positive or negative appraisal of a nation's past performance. Outputs will include classifications like objective or subjective. The class mentioned first in the response will serve as the prediction, with the class label extracted from the text between the markers &lt;final_answer&gt; and &lt;/final_answer&gt;.\n\nInput:\nOver several decades, Prime Central London – or PCL – had become a repository for cash from wealthy foreigners, whether they actually wanted to live there or not.\n\nOutput:\n&lt;final_answer&gt;objective&lt;/final_answer&gt;\n\nInput:\nFaced with a tighter labor market, many districts are raising base salaries and offering signing and relocation bonuses — up to a whopping $25,000 in one New Mexico school district.\n\nOutput:\n&lt;final_answer&gt;objective&lt;/final_answer&gt;\n\nInput:\nThat when liquidation of commodities and securities has gone too far it becomes the business of government to stop it, using public credit by such means as it may think fit.\n\nOutput:\n&lt;final_answer&gt;subjective&lt;/final_answer&gt;\n\nInput:</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Classify a given sentence as either "objective" or "subjective" based on its linguistic characteristics. Determine whether the sentence presents neutral information or expresses a personal opinion/bias. If the text maintains a detached tone, focusing on verifiable facts and data, assign the label "objective". Conversely, if the tone is evaluative, emotive, or reveals a bias, categorize it as "subjective". Compare the tone of a factual text discussing political events to a text expressing a clear opinion about a historical event to grasp the distinction between the two genres. The predicted class will be the first class mentioned in the language model's response, enclosed within the marks &lt;final_answer&gt; and &lt;/final_answer&gt;.\n\nInput:\n“This latest rule will open our borders even more, and the Court seems to relish making arbitrary decisions without thinking about consequences.\n\nOutput:\n&lt;final_answer&gt;objective&lt;/final_answer&gt;\n\nInput:\nTransportation Secretary Pete Buttigieg confirmed to The Associated Press on Thursday that $104.6 million in federal funds coming from last year’s bipartisan infrastructure bill will go toward a plan to dismantle Interstate 375, a highway built to bisect Detroit’s Black Bottom neighborhood and its epicenter of Black business, Paradise Valley.\n\nOutput:\n&lt;final_answer&gt;objective&lt;/final_answer&gt;\n\nInput:\nThe last may go very deep.\n\nOutput:\n&lt;final_answer&gt;objective&lt;/final_answer&gt;\n\nInput:</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Given a sentence, classify it as either "objective" or "subjective" based on its tone and language, considering the presence of third-person pronouns, neutral language, and opinions. Classify the output as "objective" if the tone is neutral and detached, focusing on facts and data, or as "subjective" if the tone is evaluative, emotive, or biased.\n\nInput:\nTransportation Secretary Pete Buttigieg confirmed to The Associated Press on Thursday that $104.6 million in federal funds coming from last year’s bipartisan infrastructure bill will go toward a plan to dismantle Interstate 375, a highway built to bisect Detroit’s Black Bottom neighborhood and its epicenter of Black business, Paradise Valley.\n\nOutput:\n&lt;final_answer&gt;objective&lt;/final_answer&gt;\n\nInput:\n“This latest rule will open our borders even more, and the Court seems to relish making arbitrary decisions without thinking about consequences.\n\nOutput:\n&lt;final_answer&gt;objective&lt;/final_answer&gt;\n\nInput:\nHe is fairly secure.\n\nOutput:\n&lt;final_answer&gt;objective&lt;/final_answer&gt;\n\nInput:\nIn a recent report on the “new poor,” made by the Welfare Council of New York City, there is a reference to “the mental infection of dependency.” This was upon the investigation of unemployment relief.\n\nOutput:\n&lt;final_answer&gt;objective&lt;/final_answer&gt;\n\nInput:</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Classify each sentence as objective or subjective by recognizing its language characteristics. Identify whether each sentence presents neutral information or expresses a personal opinion. If the sentence provides factual information without taking a bias, classify it as objective. Conversely, if the sentence conveys the author's perspective, emotions, or beliefs, label it as subjective. As our language model expert, carefully analyze each sentence, extracting its tone, and determine whether it presents verifiable data or the author's biased thoughts. For instance, compare a factual news report on politics to a blog post about a historical event and highlight the differences between objective and subjective writing. Our output will be the predicted class enclosed within the markers &lt;final_answer&gt; and &lt;/final_answer&gt;, with the first-mentioned class being the predicted label.\n\nInput:\n“This latest rule will open our borders even more, and the Court seems to relish making arbitrary decisions without thinking about consequences.\n\nOutput:\n&lt;final_answer&gt;objective&lt;/final_answer&gt;\n\nInput:</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Categorize the text as either objective or subjective, considering whether it presents neutral information or expresses a personal opinion/bias.\n\nObjective: The text has a neutral tone and presents factual information about the actions of Democrats in Congress and the union's negotiations.\n\nSubjective: The text has a evaluative tone and expresses a positive/negative opinion/evaluation about the past performance of the country.\n\nInput:\nOver several decades, Prime Central London – or PCL – had become a repository for cash from wealthy foreigners, whether they actually wanted to live there or not.\n\nOutput:\n&lt;final_answer&gt;objective&lt;/final_answer&gt;\n\nInput:\nThe last may go very deep.\n\nOutput:\n&lt;final_answer&gt;objective&lt;/final_answer&gt;\n\nInput:\nThat when liquidation of commodities and securities has gone too far it becomes the business of government to stop it, using public credit by such means as it may think fit.\n\nOutput:\n&lt;final_answer&gt;subjective&lt;/final_answer&gt;\n\nInput:\nThat is what it means to sell bonds.\n\nOutput:\n&lt;final_answer&gt;objective&lt;/final_answer&gt;\n\nInput:</td>
      <td>0.66</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Classify a statement as either "subjective" or "objective" based on whether it reflects a personal opinion or a verifiable fact. The output classes to include are "objective" and "subjective".\n\nInput:\nThe promotion of it for many is an avocation, for increasing numbers it is a profession, and for a very great number of more or less trained men and women it is employment and livelihood.\n\nOutput:\n&lt;final_answer&gt;objective&lt;/final_answer&gt;\n\nInput:</td>
      <td>0.65</td>
    </tr>
    <tr>
      <th>9</th>
      <td>A labeling exercise necessitates scrutinizing provided text to classify them as either vastly personal ('subjective') or dispassionately factual ('objective') based on the presence of opinions, biases, or verifiable information. Your mission is to accurately determine whether the supplied sentence leans more towards subjective expression of personal thought or objective presentation of facts, then output the corresponding classification within the format "&lt;final_answer&gt;&lt;output class&gt;, &lt;output class&gt;&lt;/final_answer&gt;" (e.g. "&lt;final_answer&gt;objective&lt;/final_answer&gt;"). Recognize that subjective sentences usually embody the writer's own views or emotions, whereas objective sentences present data without personal investment or allegiance. The predicted outcome will be the one first mentioned in the response, and the extracted class label will be positioned between the markers &lt;final_answer&gt; and &lt;/final_answer&gt;, which can only be one of the two categories: subjective or objective.\n\nInput:\nThe last may go very deep.\n\nOutput:\n&lt;final_answer&gt;objective&lt;/final_answer&gt;\n\nInput:</td>
      <td>0.65</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Classify a collection of labeled sentences as either based on fact or reflecting personal opinion, using linguistic features to distinguish between objective statements presenting verifiable information and subjective expressions of opinion or attitude, with the objective class being denoted by &lt;final_answer&gt;objective&lt;/final_answer&gt; and the subjective class by &lt;final_answer&gt;subjective&lt;/final_answer&gt;, where the first-mentioned class in the response will serve as the predicted outcome.\n\nInput:\nThe principal reason, from the point of view of government, is that a universal income tax would be a powerful restraint upon the expansion of government.\n\nOutput:\n&lt;final_answer&gt;objective&lt;/final_answer&gt;\n\nInput:</td>
      <td>0.64</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Given a dataset of sentences, use linguistic analysis to categorize each sentence as either 'objective' or 'subjective', reflecting its tone and language usage. Examine the presence of neutral third-person pronouns, factual data, and opinions to determine whether a sentence presents information in a detached and neutral manner ('objective') or conveys a personal perspective or emotional appeal ('subjective'). Your primary consideration should be the sentence's intention, purpose, and emotional resonance, with the predicted classification appearing first in your response. The predicted classification will be extracted from the text situated between the '&lt;final_answer&gt;' and '&lt;/final_answer&gt;' markers.\n\nInput:\nCOVID is continually evolving to become more immune evasive, according to Ray, and Omicron is spawning exponentially.\n\nOutput:\n&lt;final_answer&gt;objective&lt;/final_answer&gt;\n\nInput:\nThe last may go very deep.\n\nOutput:\n&lt;final_answer&gt;objective&lt;/final_answer&gt;\n\nInput:\nOver several decades, Prime Central London – or PCL – had become a repository for cash from wealthy foreigners, whether they actually wanted to live there or not.\n\nOutput:\n&lt;final_answer&gt;objective&lt;/final_answer&gt;\n\nInput:</td>
      <td>0.59</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
