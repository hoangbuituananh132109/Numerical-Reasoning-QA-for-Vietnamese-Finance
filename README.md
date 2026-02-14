# Numerical Reasoning QA for Vietnamese Financial

This repository contains source code, experimental notebooks, and datasets for the research project **Numerical Reasoning Question Answering (NumQA)** on Vietnamese financial texts, using **Small Language Models (≤14B parameters)** under limited computational resources.

The project focuses on evaluating and improving **verifiable numerical reasoning (program-based reasoning)** of language models when working with mixed **textual and tabular financial data**.

---

## Project Overview

Unlike conventional QA tasks, NumQA requires models to:

1. Understand financial context from **text and numerical tables**
2. Generate a **mathematical reasoning program** describing the computation process
3. Execute the program using an **external execution engine** to obtain the final answer

This approach provides:

- Transparent reasoning processes
- Reduced numerical errors from direct LLM computation
- Better verifiability for financial applications

This repository implements and evaluates:

- Structured prompt engineering
- Two-step self-evaluation (reasoning error detection and correction)
- Fine-tuning Phi-4 on LLM-cleaned datasets

---

## Dataset – ViNumQA

### Data Source

The dataset used is **ViNumQA**, a Vietnamese NumQA dataset released in the **VLSP 2025 – Numerical Reasoning Question Answering shared task**.

- Includes translated and verified samples from FinQA
- Augmented with newly extracted data from Vietnamese enterprise financial reports (2020–2025)
- Each sample is designed following a **program-based reasoning** paradigm

📌 The dataset can be:

- Placed directly under the `data/` directory
- Or downloaded from the official VLSP 2025 competition release

---

### Data Format

All **train / validation / test** splits share the **same JSON structure**, where each sample has the form:

```
{
  "pre_text": [...],
  "table": [...],
  "post_text": [...],
  "id": "...",
  "qa": {
    "question": "...",
    "program": "...",
    "exe_ans": "..."
  }
}
```

#### Field Description

- **pre_text**: Textual context before the table
- **table**: Financial data table (first row as headers)
- **post_text**: Textual context after the table
- **id**: Unique sample identifier
- **qa.question**: Vietnamese question
- **qa.program**: Mathematical reasoning program (available in train/val)
- **qa.exe_ans**: Execution result of the program (available in train/val)

📌 **Important:**  
During inference, the **LLM is NOT required to generate `exe_ans`**.  
The LLM **only generates the `program`**, while numerical computation is handled by an **external executor**.

---

### Program Accuracy (PA) and Execution Accuracy (EA)

Two evaluation metrics are used:

#### Program Accuracy (PA)

- Compares the generated program with the gold program
- Checks **mathematical equivalence**, allowing:
  - Commutative operation reordering (e.g., `add(a,b)` ≡ `add(b,a)`)

#### Execution Accuracy (EA)

- Executes the generated program
- Compares the final result with `exe_ans`
- **Correct PA ⇒ EA is always correct**
- **Incorrect PA ⇒ EA may still be correct**, as multiple programs can yield the same result

Run python executor/pa_ea_calculator.py to test the executor interactively.

This evaluation reflects real-world reasoning quality:

> What matters is not producing the same program, but producing the **correct computation**.

---

## Environment Setup (Google Colab – A100)

The repository is designed to run on **Google Colab (A100)** using Unsloth and vLLM.

### Library Installation

Run the following cell in a notebook:

```
%%capture
import os, re
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    import torch
    v = re.match(r"[0-9]{1,}\\.[0-9]{1,}", str(torch.__version__)).group(0)
    xformers = "xformers==" + (
        "0.0.33.post1" if v=="2.9" else
        "0.0.32.post2" if v=="2.8" else
        "0.0.29.post3"
    )
    !pip install --no-deps bitsandbytes accelerate {xformers} peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth

!pip install transformers==4.56.2
!pip install --no-deps trl==0.22.2
!pip install vllm
```

---

## Experimental Pipeline

### Prompting + Self-Evaluation (Inference-only)

- Models with ≤14B parameters (Qwen3-8B, Phi-4, Mistral-7B, etc.)
- Structured prompts specifying:
  - Task description
  - Allowed operations
  - Keyword-to-operation mapping
  - Fixed output format

#### Two-step self-evaluation:

1. **Step 1**: Generate reasoning and program
2. **Step 2**: Feed the output back into the model to detect and correct errors

This pipeline addresses common mistakes such as:

- Selecting incorrect table cells
- Using incorrect operations
- Inconsistent scaling (e.g., multiplying/dividing by 100)

---

### Fine-tuning Phi-4 with Corrected Data

Fine-tuning does **not directly use the raw dataset**:

1. Run **two-step self-evaluation with Phi-4** on the training set
2. Categorize samples into:
   - Correct PA
   - Correct EA but incorrect PA
   - Fully incorrect
3. Incorrect samples are:
   - Revised using **Gemini-2.5-Flash**
   - Required to **preserve Phi-4’s original reasoning style**
4. The cleaned dataset is used to **fine-tune Phi-4 with LoRA + Unsloth**

---

## Repository Structure

```
.
├── data/
│   ├── train.json
│   ├── valid.json
│   └── test.json
│
├── notebooks/
│   ├── inference_with_diffirence_models.ipynb       # Inference test data using models with seft-evaluation
│   └── finetune_phi4.ipynb                          # SFT Phi-4 using modified data
│
├── results/
│   ├── details/                                     # Contains the question, program_step2 and gold_program of the models
│   └── evaluation_results/                          # Contains PA, EA of the models and visual charts
│   └── output_analyst.ipynb                         # Analyze data from details/ and save in evaluation_results/
│
├── data_analyst.ipynb                               # Analyze data from viNumQA
|
└── README.md
```

---

## Fast Inference Guide (vLLM)

- Inference on 497 test samples
- Uses **vLLM + fast inference** on A100
- Applies the self-evaluation pipeline

See notebook:

```
notebooks/inference_with_difference_models.ipynb
```

---

## Repository Goals

- Provide a **transparent, reproducible, and low-cost** NumQA pipeline
- Suitable for:
  - Students
  - Small research groups
  - Resource-constrained environments
- Serve as an open foundation for Vietnamese financial NLP research

---

## Results (Summary)

| Model | PA (%) | EA (%) |
|------|--------|--------|
| Llama-3.1-8B | ~0.20 | ~0.40 |
| Mistral-7B-Instruct | 38.63 | 41.65 |
| Phi-4 (self-eval) | 54.69 | 59.31 |
| Qwen3-8B (self-eval) | **59.56** | **64.19** |
| Phi-4 (SFT, Step 2) | 52.52 | 56.54 |

---

## License

MIT License

---

## Contact

Hoang Bui Tuan Anh¹, Le Thanh Dat¹, Nguyen Duc Anh¹, Tran Hong Viet¹*

¹ University of Engineering and Technology,  
Vietnam National University, Hanoi, Vietnam  
Email: {22022611, 22022627, 22022661, thviet}@vnu.edu.vn  

*Corresponding author

