# PACEvolve: Enabling Long-Horizon Progress-Aware Consistent Evolution

## Table of Contents
1. [About the Project](#about-the-project)
2. [Prerequisites](#prerequisites)
3. [Installation & Usage](#installation--usage)
4. [Support & Contribution](#support--contribution)
5. [License](#license)

---

## About the Project

This repo contains implementation for the [PACEvolve](https://arxiv.org/pdf/2601.10657) paper.

---

## Prerequisites

Before installing `PACEvolve` you need:

* **Python 3.9** or later
* We support OpenAI / Anthropic / Google Gemini APIs. You should have at least one API key available, more installation instructions can be found below.
* **Git**

The project relies on specific directory structures to locate tasks and configurations. Ensure your project root contains:
* `workflows/` (where this script resides)
* `tasks/` (containing task definitions and their respective environment installation requirements)

---

## Installation & Usage

### 1. Installation

Clone the repository and install the required dependencies.

```bash
git clone [https://github.com/MinghaoYan/PACEvolve.git](https://github.com/MinghaoYan/PACEvolve.git)
cd auto-evo
pip install -r requirements.txt
```

---

### 2. Setup API Keys & Dependencies

You must configure the API keys and install the necessary packages for the models you intend to use. We currently support Google Gemini, OpenAI, and Anthropic.

**1. Set your API Keys**
Export the environment variables for the providers you plan to use:

```bash
# For Google Gemini
export GOOGLE_API_KEY="your_gemini_key"

# For OpenAI (GPT-4)
export OPENAI_API_KEY="your_openai_key"

# For Anthropic (Claude)
export ANTHROPIC_API_KEY="your_anthropic_key"

# Install all supported clients
pip install google-generativeai openai anthropic
```

Each task contains a config.yaml file in the config subdirectory, you can change the backbone llm by changing the llm section in the config file.


### 3. Running Your First Experiment
To run the evolutionary process, execute the script with a specific task_id. This assumes you have a task configuration file located at ../tasks/<task_id>/config/.

```bash
python run_experiment.py --task_id "my_task"
```

### 4. PACEvolve++ (advisor-RL)

[PACEvolve++](https://arxiv.org/pdf/2605.07039) splits the LLM into a trainable *advisor* (which selects an idea) and a stronger *implementer* (which writes the code), and adds a search-dynamics-aware reinforcement-learning objective. Run it with the dedicated driver:

```bash
cd workflows
python run_advisor_rl.py --task_id eplb --backend mock --objective pacevolve++ --max_steps 2
```

Add `advisor_llm` / `implementer_llm` sections to the task config to give the two roles different models — e.g. a small local model for the advisor (`client_type: ollama` + `base_url` pointing at a vLLM/Ollama server) and a frontier model for the implementer. The `rl` section selects the objective (`pacevolve++`, `grpo`, `entropic`, `maxk`, `none`). The RL layer has a pytest suite:

```bash
python3 -m venv .venv && .venv/bin/pip install numpy pytest pyyaml
.venv/bin/pytest
```

---

## Support & Contribution
### Documentation
Transcript Logs: Detailed logs of the LLM's thought process and code generation are saved to the transcripts defined in your YAML config.
Controller Logs: Technical execution logs are saved to controller_verbose_*.log.

If you need help, please open an issue in the repository!

---

## License

**Open-source project**

You are free to copy, modify, and distribute `PACEvolve` with attribution under the terms of the **Apache 2.0 license**. See the `LICENSE` file for details.

---

This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

This project is intended for demonstration purposes only. It is not intended for use in a production environment.
