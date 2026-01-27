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
* **Google Gemini API Key** (for `google.generativeai`)
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

### 2. Setup API Key

You must configure your Google GenAI key. 

```bash
export GOOGLE_API_KEY="your_key"
```
---

### 3. Running Your First Experiment
To run the evolutionary process, execute the script with a specific task_id. This assumes you have a task configuration file located at ../tasks/<task_id>/config/.

```bash
python run_experiment.py --task_id "my_task"
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
