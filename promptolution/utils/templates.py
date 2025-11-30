"""Meta-prompt templates for different prompt optimization methods."""

DEFAULT_SYS_PROMPT = "You are a helpful assistant."

EVOPROMPT_DE_TEMPLATE = """Please follow the instruction step-by-step to generate a better prompt.
Identifying the different parts between Prompt 1 and Prompt 2:
Prompt 1: Your task is to classify the comment as one of the following categories: terrible, bad, okay, good, great.
Prompt 2: In this task, you are given sentences from movie reviews. The task is to classify a sentence as one of the following categories: terrible, bad, okay, good, great.
Different parts:
"Your task is to classify the comment" vs "In this task, you are given sentences from movie reviews. The task is to classify a sentence"
"comment" vs "sentences from movie reviews"

2. Randomly mutate the different parts:
"Your task is to classify the comment" -> "The objective is to categorize the statement"
"comment" -> "phrases in movie reviews"

3. Crossover the different parts with the following Prompt 3 and generate a final prompt bracketed with <prompt> and </prompt>:
Prompt 3: You are a sentiment classifier. To do this, you must first understand the meaning of the sentence and any relevant context. And then you should classify it as one of the following categories: terrible, bad, okay, good, great.

Final Prompt: <prompt>As a sentiment classifier, analyze phrases in movie reviews and categorize them into one of the following categories: terrible, bad, okay, good, great, while considering the meaning and relevant context.</prompt>

Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between the Prompt 1 and Prompt 2:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Randomly mutate the different parts
3. Crossover the different parts with the following Prompt 3 and generate a final prompt bracketed with <prompt> and </prompt>:
Prompt 3: <prompt0>

1."""

EVOPROMPT_DE_TEMPLATE_TD = """Please follow the instruction step-by-step to generate a better prompt for the following task: The dataset consists of movie reviews with five levels of sentiment labels: terrible, bad, neutral, okay, good, and great. The task is to classify each movie review into one of these five sentiment categories. The class mentioned first in the response of the LLM will be the prediction.
Identifying the different parts between Prompt 1 and Prompt 2:
Prompt 1: Your task is to classify the comment as one of the following categories: terrible, bad, okay, good, great.
Prompt 2: In this task, you are given sentences from movie reviews. The task is to classify a sentence as one of the following categories: terrible, bad, okay, good, great.
Different parts:
"Your task is to classify the comment" vs "In this task, you are given sentences from movie reviews. The task is to classify a sentence"
"comment" vs "sentences from movie reviews"

2. Randomly mutate the different parts:
"Your task is to classify the comment" -> "The objective is to categorize the statement"
"comment" -> "phrases in movie reviews"

3. Crossover the different parts with the following Prompt 3 and generate a final prompt bracketed with <prompt> and </prompt>:
Prompt 3: You are a sentiment classifier. To do this, you must first understand the meaning of the sentence and any relevant context. And then you should classify it as one of the following categories: terrible, bad, okay, good, great.

Final Prompt: <prompt>As a sentiment classifier, analyze phrases in movie reviews and categorize them into one of the following categories: terrible, bad, okay, good, great, while considering the meaning and relevant context.</prompt>

Please follow the instruction step-by-step to generate a better prompt for the following task: <task_desc>
1. Identify the different parts between the Prompt 1 and Prompt 2:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Randomly mutate the different parts
3. Crossover the different parts with the following Prompt 3 and generate a final prompt bracketed with <prompt> and </prompt>:
Prompt 3: <prompt0>

1."""

EVOPROMPT_GA_TEMPLATE = """Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. Crossover Prompt: Rewrite the complex text into simpler text while keeping its meaning.
2. <prompt>Transform the provided text into simpler language, maintaining its essence.</prompt>

Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1."""

EVOPROMPT_GA_TEMPLATE_TD = """Please follow the instruction step-by-step to generate a better prompt for the following task: The dataset consists of texts to be simplified. The meaning of the texts is to be kept.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. Crossover Prompt: Rewrite the complex text into simpler text while keeping its meaning.
2. <prompt>Transform the provided text into simpler language, maintaining its essence.</prompt>

Please follow the instruction step-by-step to generate a better prompt for the following task: <task_desc>
1. Crossover the following prompts and generate a new prompt:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1."""

OPRO_TEMPLATE = """Your task is to generate an instruction.

Below are some previous instructions with their scores. The score ranges from 0 to 100.

<instructions>
Here are some examples of the target dataset:
<examples>

Generate a new instruction bracketed with <prompt> and ending it with </prompt> that is different from all the instructions above and has a higher score than all the instructions above. The instruction should be concise, effective, and generally applicable to the task described.

Your new instruction:"""

OPRO_TEMPLATE_TD = """Your task is to generate an instruction for the following task:
<task_desc>

Below are some previous instructions with their scores. The score ranges from 0 to 100.

<instructions>
Here are some examples of the target dataset:
<examples>

Generate a new instruction bracketed with <prompt> and ending it with </prompt> that is different from all the instructions above and has a higher score than all the instructions above. The instruction should be concise, effective, and generally applicable to the task described.

Your new instruction:"""

PROMPT_VARIATION_TEMPLATE = """Generate a single variation of the following instruction while keeping the semantic meaning.
Generate the variation starting with <prompt> and ending with </prompt> tags.

Input: <prev_prompt>

Output:"""

PROMPT_CREATION_TEMPLATE = """You are asked to give the corresponding prompt that gives the following outputs given these inputs.
Return it starting with <prompt> and ending with </prompt> tags.
Include the name of the output classes in the prompt.

<input_output_pairs>

The instruction was"""

PROMPT_CREATION_TEMPLATE_TD = """You are asked to give the corresponding prompt that gives the following outputs given these inputs for the following task: <task_desc>.
Return it starting with <prompt> and ending with </prompt> tags.
Include the name of the output classes in the prompt.

<input_output_pairs>

The instruction was"""

PROMPT_CREATION_TEMPLATE_FROM_TASK_DESCRIPTION = """Please create diverse system prompts for the following task, not using any placeholders, working universally, for any datapoint-specific instructions following each system prompt.

Task: <task_desc>

Explicitly state the expected format above by repeating its exact character sequence verbatim in every prompt if applicable.

Create overall <n_prompts> prompts within json format, meaning strings inside quotations marks ("") as an array. Do not response with anything else. Start the array with [ and end with ]. Separate each prompt by a comma, and do not use quotation marks inside the prompts."""


DOWNSTREAM_TEMPLATE = "<instruction>"

DOWNSTREAM_TEMPLATE_W_FEWSHOTS = """<instruction>

<few_shots>

Input:"""

CAPO_FEWSHOT_TEMPLATE = """Input:
<input>
Output:
<output>"""

CAPO_CROSSOVER_TEMPLATE = """You receive two prompts for the following task: <task_desc>
Please merge the two prompts into a single coherent prompt. Maintain the key linguistic features from both original prompts:
Prompt 1: <mother>
Prompt 2: <father>

Return the new prompt in the following format:
<prompt>new prompt</prompt>"""

CAPO_MUTATION_TEMPLATE = """You receive a prompt for the following task: <task_desc>
Please rephrase the prompt, preserving its core meaning while substantially varying the linguistic style.
Prompt: <instruction>

Return the new prompt in the following format:
<prompt>new prompt</prompt>"""


default_prompts = [
    "Give me your response within <final_answer> tags.",
    "Please provide a thoughtful answer to my question and wrap your response in <final_answer> tags so I can easily identify it.",
    "I need your expertise on this matter. Kindly structure your response within <final_answer> tags for better readability.",
    "Analyze the following and present your findings enclosed in <final_answer> </final_answer> tags.",
    "Consider this inquiry carefully. Your comprehensive response should be formatted within <final_answer> tags to facilitate extraction.",
    "Respond succinctly. Ensure all content appears between <final_answer> and </final_answer> markers.",
    "Would you mind addressing this request? Please place your entire response inside <final_answer> </final_answer> formatting.",
    "I'm seeking your insights on a particular topic. Kindly ensure that your complete analysis is contained within <final_answer> tags for my convenience.",
    "Examine this query thoroughly and deliver your conclusions. All output must be encapsulated in <final_answer> </final_answer> notation for processing purposes.",
    "Help me understand this subject better. Your explanation should begin with <final_answer> and conclude with </final_answer> to maintain proper structure.",
    "I require information on the following. Please format your response with <final_answer> tags at the beginning and end for clarity.",
    "Contemplate this scenario and offer your perspective. Remember to enclose all content within <final_answer> tags as per requirements.",
    "Elaborate on this concept, making sure to wrap the entirety of your explanation in <final_answer> </final_answer> markers for systematic review.",
    "Describe your approach to this situation. Be thorough yet concise, and place your complete response between <final_answer> and </final_answer> tags.",
    "Share your knowledge on this matter. Your entire response should be presented within <final_answer> tags to facilitate proper integration into my workflow.",
    "Let's think step by step. Your answer should be enclosed within <final_answer> </final_answer> tags.",
    "Provide a detailed response to the following question, ensuring that all information is contained within <final_answer> tags for easy extraction.",
    "Kindly address the following topic, formatting your entire response between <final_answer> and </final_answer> markers for clarity and organization.",
    "Offer your insights on this issue, making sure to encapsulate your full response within <final_answer> tags for seamless processing.",
    "Delve into this subject and present your findings, ensuring that all content is wrapped in <final_answer> </final_answer> notation for systematic analysis.",
    "Illuminate this topic with your expertise, formatting your complete explanation within <final_answer> tags for straightforward comprehension.",
    "Provide your perspective on this matter, ensuring that your entire response is contained within <final_answer> tags for efficient review.",
    "Analyze the following scenario and deliver your conclusions, making sure to enclose all output in <final_answer> </final_answer> markers for clarity.",
    "Help me grasp this concept better by structuring your explanation between <final_answer> and </final_answer> tags for proper formatting.",
]
