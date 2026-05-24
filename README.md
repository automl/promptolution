
![Coverage](https://img.shields.io/badge/Coverage-96%25-brightgreen)
[![CI](https://github.com/automl/promptolution/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/automl/promptolution/actions/workflows/ci.yml)
[![Docs](https://github.com/automl/promptolution/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/automl/promptolution/actions/workflows/docs.yml)
[![PyPI version](https://img.shields.io/pypi/v/promptolution.svg)](https://pypi.org/project/promptolution/)
![Code Style](https://img.shields.io/badge/Code%20Style-black-black)
![Python Versions](https://img.shields.io/badge/Python%20Versions-%E2%89%A53.10-blue)
[![Getting Started](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/automl/promptolution/blob/main/tutorials/getting_started.ipynb)

![promptolution](https://github.com/user-attachments/assets/84c050bd-61a1-4f2e-bc4e-874d9b4a69af)

<p align="center">
<img height="60" alt="lmu_logo" src="https://github.com/user-attachments/assets/5aecd0d6-fc2d-48b2-b395-d1877578a3c5" />
<img height="60" alt="mcml" src="https://github.com/user-attachments/assets/d9f3b18e-a5ec-4c3f-b449-e57cb977f483" />
<img height="60" alt="ellis_logo" src="https://github.com/user-attachments/assets/60654a27-0f8f-4624-a1d5-5122f2632bec" />
<img height="60" alt="uni_freiburg_color" src="https://github.com/user-attachments/assets/f5eabbd2-ae6a-497b-857b-71958ed77335" />
<img height="60" alt="tum_logo" src="https://github.com/user-attachments/assets/982ec2f0-ec14-4dc2-8d75-bfae09d4fa73" />
</p>

## 🚀 What is Promptolution?

**Promptolution** is a unified, modular framework for prompt optimization built for researchers and advanced practitioners who want full control over their experimental setup. Unlike end-to-end application frameworks with high abstraction, promptolution focuses exclusively on the optimization stage, providing a clean, transparent, and extensible API. It allows for simple prompt optimization for one task up to large-scale reproducible benchmark experiments. 

<img width="808" height="356" alt="promptolution_framework" src="https://github.com/user-attachments/assets/e3d05493-30e3-4464-b0d6-1d3e3085f575" />

### Key Features

* Implementation of many current prompt optimizers out of the box.
* Unified LLM backend supporting API-based models, Local LLMs, and vLLM clusters.
* Built-in response caching to save costs and parallelized inference for speed.
* Detailed logging and token usage tracking for granular post-hoc analysis.

Have a look at our [Release Notes](https://automl.github.io/promptolution/release-notes/) for the latest updates to promptolution.

## 📚 Scientific Publications Powered by Promptolution

- **CANTANTE: Optimizing Agentic Systems via Contrastive Credit Attribution** — Zehle, 2026. [arXiv](https://arxiv.org/abs/2605.13295)
- **MO-CAPO: Multi-Objective Cost-Aware Prompt Optimization** — Büssing et al., 2026. [arXiv](https://arxiv.org/abs/2605.18869)
- **promptolution: A Unified, Modular Framework for Prompt Optimization** — Zehle et al., 2026. [EACL 2026](https://aclanthology.org/2026.eacl-demo.21/)
- **Can Calibration of Positional Encodings Enhance Long Context Utilization?** — Zehle & Aßenmacher, 2026. [EACL 2026](https://aclanthology.org/2026.findings-eacl.120/)
- **Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky** — Hathidara et al., 2025. [arXiv](https://arxiv.org/abs/2507.03336)
- **CAPO: Cost-Aware Prompt Optimization** — Zehle et al., 2025. [AutoML 2025](https://proceedings.mlr.press/v293/zehle25a.html)

## 🔧 Installation and Quickstart

```
pip install promptolution[api]
```

For local inference, add `[transformers]` (HuggingFace) or `[vllm]` (vLLM serving), or both.

```python
import pandas as pd
from promptolution.utils import ExperimentConfig
from promptolution.helpers import run_experiment

# DataFrame with columns "x" (input) and "y" (label)
df = pd.read_csv("your_data.csv")

config = ExperimentConfig(
    optimizer="capo",
    task_description="Classify each sentence as subjective or objective.",
    prompts=["Classify the text as objective or subjective."],
    n_steps=10,
    api_url="https://api.openai.com/v1",
    model_id="gpt-4o-mini",
    api_key="YOUR_API_KEY",
)

best_prompts = run_experiment(df, config)
print(best_prompts)
```

Full tutorial: [Getting Started notebook](https://github.com/automl/promptolution/blob/main/tutorials/getting_started.ipynb) · [Docs](https://automl.github.io/promptolution/)


## 🧠 Featured Optimizers

| **Name**      | **Paper**                                              | **Init prompts** | **Exploration** | **Costs** | **Parallelizable** | **Few-shot** |
| ---- | ---- | ---- |----  |----  |  ----|----  |
| `CAPO`        | [Zehle et al., 2025](https://openreview.net/forum?id=UweaRrg9D0) | required         | 👍              | 💲        | ✅                  | ✅            |
| `EvoPromptDE` | [Guo et al., 2023](https://openreview.net/forum?id=ZG3RaNIsO8)   | required         | 👍              | 💲💲      | ✅                  | ❌            |
| `EvoPromptGA` | [Guo et al., 2023](https://openreview.net/forum?id=ZG3RaNIsO8)   | required         | 👍              | 💲💲      | ✅                  | ❌            |
| `OPRO`        | [Yang et al., 2023](https://openreview.net/forum?id=Bb4VGOWELI)  | optional         | 👎              | 💲💲      | ❌                  | ❌            |


## 🏗 Components

* **`Task`** – Manages the dataset, evaluation metrics, and subsampling.
* **`Predictor`** – Defines how to extract the answer from the model's response.
* **`LLM`** – A unified interface handling inference, token counting, and concurrency.
* **`Optimizer`** – The core component that implements the algorithms that refine prompts.
* **`ExperimentConfig`** – A configuration abstraction to streamline and parametrize large-scale scientific experiments.

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow, code quality guidelines, and how to run tests.


## 📄 Citation

If you use Promptolution in your research, please cite:

```bibtex
@inproceedings{zehle2026promptolution,
  title={promptolution: A unified, modular framework for prompt optimization},
  author={Zehle, Tom and Hei{\ss}, Timo and Schlager, Moritz and A{\ss}enmacher, Matthias and Feurer, Matthias},
  booktitle={Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 3: System Demonstrations)},
  pages={282--296},
  year={2026}
}
```

---

Developed    by **Timo Heiß**, **Moritz Schlager**, **Tom Zehle**, and **Henri Oberpaur** (LMU Munich, MCML, ELLIS, TUM, Uni Freiburg).