
![Coverage](https://img.shields.io/badge/Coverage-91%25-brightgreen)
[![CI](https://github.com/finitearth/promptolution/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/finitearth/promptolution/actions/workflows/ci.yml)
[![Docs](https://github.com/finitearth/promptolution/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/finitearth/promptolution/actions/workflows/docs.yml)
![Code Style](https://img.shields.io/badge/Code%20Style-black-black)
![Python Versions](https://img.shields.io/badge/Python%20Versions-≥3.10-blue)
[![Getting Started](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/finitearth/promptolution/blob/main/tutorials/getting_started.ipynb)

![promptolution](https://github.com/user-attachments/assets/84c050bd-61a1-4f2e-bc4e-874d9b4a69af)



<p align="center">
  <img src="https://github.com/user-attachments/assets/41edb678-16c4-4a3a-a579-62459106cf48" height="40">
  <img src="https://github.com/user-attachments/assets/1ae42b4a-163e-43ed-b691-c253d4f4c814"  height="40">
  <img  src="https://github.com/user-attachments/assets/d6977ca8-7b52-4714-83d7-610743c7d52c"  height="40"/>
  <img src="https://mcml.ai/images/footer/lmu_white.webp" height="40">
  <img src="https://mcml.ai/images/footer/tum_white.webp" height="40">
</p>


## 🚀 What is Promptolution?

**Promptolution** is a unified, modular framework for prompt optimization — built for researchers and advanced practitioners who want full control over their experimental setup. Unlike end-to-end application frameworks with high abstraction, promptolution focuses exclusively on the optimization stage, providing a clean, transparent, and extensible API. It allows for simple prompt optimization for one task up to large-scale reproducible benchmark experiments. 

<img width="808" height="356" alt="promptolution_framework" src="https://github.com/user-attachments/assets/e3d05493-30e3-4464-b0d6-1d3e3085f575" />

### Key Features

* Implementation of many current prompt optimizers out of the box.
* Unified LLM backend supporting API-based models, Local LLMs, and vLLM clusters.
* Built-in response caching to save costs and parallelized inference for speed.
* Detailed logging and token usage tracking for granular post-hoc analysis.

Have a look at our [Release Notes](https://finitearth.github.io/promptolution/release-notes/) for the latest updates to promptolution.

## 📦 Installation

```
pip install promptolution[api]
```

Local inference via vLLM or transformers:

```
pip install promptolution[vllm,transformers]
```

From source:

```
git clone https://github.com/finitearth/promptolution.git
cd promptolution
poetry install
```

## 🔧 Quickstart

Start with the **Getting Started tutorial**:
[https://github.com/finitearth/promptolution/blob/main/tutorials/getting_started.ipynb](https://github.com/finitearth/promptolution/blob/main/tutorials/getting_started.ipynb)

Full docs:
[https://finitearth.github.io/promptolution/](https://finitearth.github.io/promptolution/)


## 🧠 Featured Optimizers

| **Name**      | **Paper**                                              | **Init prompts** | **Exploration** | **Costs** | **Parallelizable** | **Few-shot** |
| ---- | ---- | ---- |----  |----  |  ----|----  |
| `CAPO`        | [Zehle et al., 2025](https://arxiv.org/abs/2504.16005) | required         | 👍              | 💲        | ✅                  | ✅            |
| `EvoPromptDE` | [Guo et al., 2023](https://arxiv.org/abs/2309.08532)   | required         | 👍              | 💲💲      | ✅                  | ❌            |
| `EvoPromptGA` | [Guo et al., 2023](https://arxiv.org/abs/2309.08532)   | required         | 👍              | 💲💲      | ✅                  | ❌            |
| `OPRO`        | [Yang et al., 2023](https://arxiv.org/abs/2309.03409)  | optional         | 👎              | 💲💲      | ❌                  | ❌            |

## 🏗 Components

* **`Task`** – Manages the dataset, evaluation metrics, and subsampling.
* **`Predictor`** – Defines how to extract the answer from the model's response.
* **`LLM`** – A unified interface handling inference, token counting, and concurrency.
* **`Optimizer`** – The core component that implements the algorithms that refine prompts.
* **`ExperimentConfig`** – A configuration abstraction to streamline and parametrize large-scale scientific experiments.

## 🤝 Contributing

Open an issue → create a branch → PR → CI → review → merge.
Branch naming: `feature/...`, `fix/...`, `chore/...`, `refactor/...`.

Please ensure to use pre-commit, which assists with keeping the code quality high:

```
pre-commit install
pre-commit run --all-files
```
We encourage every contributor to also write tests, that automatically check if the implementation works as expected:

```
poetry run python -m coverage run -m pytest
poetry run python -m coverage report
```

Developed by **Timo Heiß**, **Moritz Schlager**, and **Tom Zehle** (LMU Munich, MCML, ELLIS, TUM, Uni Freiburg).
