
![Coverage](https://img.shields.io/badge/Coverage-91%25-brightgreen)
[![CI](https://github.com/finitearth/promptolution/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/finitearth/promptolution/actions/workflows/ci.yml)
[![Docs](https://github.com/finitearth/promptolution/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/finitearth/promptolution/actions/workflows/docs.yml)
![Code Style](https://img.shields.io/badge/Code%20Style-black-black)
![Python Versions](https://img.shields.io/badge/Python%20Versions-≥3.10-blue)
[![Getting Started](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/finitearth/promptolution/blob/main/tutorials/getting_started.ipynb)

![promptolution](https://github.com/user-attachments/assets/84c050bd-61a1-4f2e-bc4e-874d9b4a69af)



<p align="center">
  <img src="https://mcml.ai/images/MCML_Logo_cropped.jpg" height="45">
  <img src="https://github.com/user-attachments/assets/1ae42b4a-163e-43ed-b691-c253d4f4c814"  height="45">
  <img  src="https://github.com/user-attachments/assets/e70ec1d4-bbc4-4ff3-8803-8806bc879bb0"  height="45"/>
  <img src="https://mcml.ai/images/footer/lmu_white.webp" height="45">
  <img src="https://mcml.ai/images/footer/tum_white.webp" height="45">
</p>



## 🚀 What is Promptolution?

**Promptolution** is a modular framework for *serious* prompt optimization — built for researchers who want full control over optimizers, datasets, evaluation, and logging.
Unlike end-to-end agent frameworks (DSPy, LangGraph…), Promptolution focuses **exclusively** on the prompt optimization phase, with clean abstractions, transparent internals, and an extensible API.

It supports:

* single-task prompt optimization
* large-scale experiments
* local + API-based LLMs
* fast parallelization
* clean logs for reproducible research

Developed by **Timo Heiß**, **Moritz Schlager**, and **Tom Zehle** (LMU Munich, MCML, ELLIS, TUM, Uni Freiburg).



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



## 🏗 Core Components

* **Task** – wraps dataset fields, init prompts, evaluation.
* **Predictor** – runs predictions using your LLM backend.
* **LLM** – unified interface for OpenAI, HuggingFace, vLLM, etc.
* **Optimizer** – plug-and-play implementations of CAPO, GA/DE, OPRO, and your own custom ones.



## ⭐ Highlights

* Modular, OOP design → easy customization
* Experiment-ready architecture
* Parallel LLM requests
* LangChain support
* JSONL logging, callbacks, detailed event traces
* Works from laptop to cluster



## 📜 Changelog

[https://finitearth.github.io/promptolution/release-notes/](https://finitearth.github.io/promptolution/release-notes/)



## 🤝 Contributing

Open an issue → create a branch → PR → CI → review → merge.
Branch naming: `feature/...`, `fix/...`, `chore/...`, `refactor/...`.

### Code Style

```
pre-commit install
pre-commit run --all-files
```

### Tests

```
poetry run python -m coverage run -m pytest
poetry run python -m coverage report
```
Just tell me — happy to tune it further.
