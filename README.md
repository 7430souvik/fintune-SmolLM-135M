# SmolLM-135M Financial Domain Adaptation

> Continued pre-training of SmolLM-135M on SEC 10-K financial
> filings using LoRA, Unsloth and 4-bit quantization.

## 🚀 Overview

This project explores how a small language model can be adapted
to a specialized financial domain using Continued Pre-Training
(CPT).

The base `HuggingFaceTB/SmolLM-135M` model was trained on SEC
10-K annual filings using LoRA and Unsloth on a Google Colab
Tesla T4 GPU.

The experiment evaluates whether the model learns financial
language through:

- Validation perplexity
- Before/after generation comparison
- General English evaluation
- Catastrophic forgetting analysis

## 📊 Key Result

| Metric | Base Model | After CPT |
|---|---:|---:|
| SEC Validation Perplexity | 19.1 | **13.2** |

**31.0% reduction in validation perplexity**

The evaluation was performed on the same held-out SEC validation
chunks before and after continued pre-training.

## 🧠 Model

- Base Model: `HuggingFaceTB/SmolLM-135M`
- Parameters: ~135M-class small language model
- Quantization: 4-bit
- Fine-tuning: LoRA
- Framework: Unsloth + Hugging Face
- Hardware: Tesla T4
- Environment: Google Colab

## 📚 Dataset

SEC 10-K annual filings from the `PleIAs/SEC` dataset.

### Dataset preparation

- 100 filings collected
- 80 filings for training
- 20 filings for validation
- Filing-level train/validation split
- 6,830 training chunks
- 1,191 validation chunks
- 256-word chunks
- 20% overlap

The split was performed before chunking to reduce the risk of
data leakage.

## ⚙️ Training

- Epochs: 2
- Effective batch size: 32
- Learning rate: 2e-4
- Embedding learning rate: 2e-5
- LoRA rank: 32
- LoRA alpha: 32
- Optimizer: AdamW 8-bit
- Sequence length: 512
- Gradient checkpointing: enabled

## 🔬 Evaluation

### 1. Domain Perplexity

The model's perplexity on held-out SEC financial text decreased
from 19.1 to 13.2.

### 2. Generation

The same financial prompts were evaluated before and after CPT
to inspect changes in financial language generation.

### 3. Catastrophic Forgetting

General English prompts were evaluated after CPT to investigate
whether specialization affected general language behavior.

## 📓 Google Colab

The complete experiment was developed and executed in Google
Colab.

See:

`notebooks/financial_cpt.ipynb`

## 🛠️ Tech Stack

Python • PyTorch • Hugging Face Transformers • PEFT •
Unsloth • Datasets • Google Colab • CUDA
