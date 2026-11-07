# AI Use Documentation

We used Claude and ChatGPT during different parts of our DSC 180A project. The following sections summarize our main AI interactions, including what we asked, what AI helped us with, and how we checked the results.

## 1. AI Contributions: Programming

### Task Generator

- **Asked:** We asked Claude to help develop a Python task generator that starts with a random integer and creates a sequence of dependent arithmetic operations. Each step should use the result from the previous step, and the program should calculate the expected answers automatically.
- **AI Output:** Claude assisted with implementing `tasks.py`, including random task generation, arithmetic operations, and expected-answer calculations.
- **Checked:** We reviewed the task-generation logic and compared the generated operations with their expected answers. The code also includes a self-test for checking the scoring logic.

### Experiment Pipeline

- **Asked:** We asked Claude to help build an experiment pipeline that compares three prompting methods: bulk, multiturn, and chained. We wanted to run the same tasks under each condition and record model responses, accuracy, and token usage.
- **AI Output:** Claude assisted with implementing the experiment runner and scoring functions in `run_experiment.py` and `tasks.py`. The pipeline uses Ollama to run Qwen3.5:4B locally and saves experimental records in JSONL format.
- **Checked:** We compared the recorded accuracy results across the three prompting conditions. The pipeline also provides a dry-run option for testing without calling the model.

## 2. AI Contributions: Report

### Literature Review and Introduction

- **Asked:** We provided ChatGPT with our research question, project background, and related papers, and asked it to help organize our ideas into English paragraphs and improve our literature review.
- **AI Output:** ChatGPT helped draft and revise parts of the Introduction and Prior Work sections, improve grammar, and organize the explanations of how previous research relates to our experiment.
- **Checked:** We reviewed the text against our research question and the referenced papers, and checked that the citations matched the sources used in our report.

### Experimental Methods and Data Description

- **Asked:** We explained our task-generation process and three prompting conditions to ChatGPT and asked it to help describe our methods more clearly in English.
- **AI Output:** ChatGPT helped revise the Data and Tasks section and explain our expansion from 30 to 100 tasks. It also assisted with LaTeX formatting and improving sentence clarity.
- **Checked:** We compared the descriptions with our Python implementation and experimental setup.

### Results and Discussion

- **Asked:** We provided ChatGPT with our experimental results from 30 and 100 tasks, including the accuracy comparison and accuracy-by-step figure. We asked it to help organize the findings, discuss possible explanations, and avoid repetitive writing.
- **AI Output:** ChatGPT helped draft and revise parts of the Abstract, Results, and Discussion. It also helped explain the differences between prompting methods and identify limitations of our experiment.
- **Checked:** We compared the written results with our reported accuracy values and figures. We also reviewed the explanations to avoid making conclusions that were not supported by our experiments.

## 3. Human Contributions

Our team provided the research question, experimental requirements, and results used throughout the project. We made decisions about the prompting conditions, conducted the experiments, and interpreted the findings.

AI tools assisted with code development and report writing, while our team reviewed the generated materials and made the final decisions about what to include.

## 4. Testing and Verification

Our project includes a self-test for checking the task generator and scoring logic, as well as a dry-run option for checking the experiment pipeline without running the model.

The relevant commands are:

```bash
python tasks.py selftest
python run_experiment.py --dry-run --n 2 --out dry.jsonl
```
