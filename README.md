# 🦙 LLM Fine-Tuning Pipeline — LLaMA 3.2-1B with QLoRA + DPO

> A complete, end-to-end Large Language Model fine-tuning pipeline built entirely on free GPUs (Kaggle + Google Colab). This project covers everything from supervised fine-tuning to preference alignment, evaluation, and live deployment.

**[Live Demo](https://huggingface.co/spaces/pranav6905/llm-finetuning-pipeline)** &nbsp;|&nbsp;
**[SFT Adapters](https://huggingface.co/pranav6905/Llama-3.2-1B-SFT-DPOMix-Adapters)** &nbsp;|&nbsp;
**[Final Model](https://huggingface.co/pranav6905/llama-1b-sft-dpo-final)** &nbsp;|&nbsp;
**[Dataset](https://huggingface.co/datasets/argilla/dpo-mix-7k)**

---

## What is Fine-Tuning? (Simple Explanation)

Imagine you hire a very smart person who has read the entire internet (that's the base LLM). They know a lot about everything, but they speak in a very generic way and sometimes give unhelpful or wrong answers.

**Fine-tuning** is like giving that person a specific training course — you show them thousands of examples of good conversations and tell them "this is how I want you to respond." After training, they keep all their general knowledge but now respond in a much better, more helpful way.

This project does that in **two stages:**

```
Base LLaMA 3.2-1B
      ↓
Stage 1: SFT — "Learn what a good response looks like"
      ↓
Stage 2: DPO — "Learn to prefer better responses over worse ones"
      ↓
Fine-Tuned & Aligned Model 🎯
```

---

## What are LoRA Adapters? (Why we don't save the full model every time)

A base LLaMA 3.2-1B model is **2.5 GB**. If you fine-tune it 10 times for 10 different tasks, you'd need to store 25 GB of nearly identical model copies. That's wasteful.

**LoRA (Low-Rank Adaptation)** solves this elegantly. Instead of modifying the entire model, it trains a tiny set of extra weights called **adapters** — think of them like a small plugin that sits on top of the base model.

```
Base Model (2.5 GB, frozen, never changes)
     +
LoRA Adapters (~50 MB, task-specific, swappable)
     =
Fine-Tuned Behavior ✅
```

The magic is that you can:
- Store only the 50 MB adapters instead of the full 2.5 GB model
- Swap adapters instantly for different tasks
- Remove adapters when done and get your base model back
- Combine multiple adapters for multi-task behavior

In this project we saved adapters at each stage — **SFT adapters** after Stage 1, and **DPO adapters** after Stage 2 — keeping things modular and storage-efficient.

---

## Why QLoRA? (Making Fine-Tuning Accessible)

Normal fine-tuning requires expensive GPUs with 40–80 GB VRAM. Most people (and students) don't have that. **QLoRA** makes it possible on free hardware.

Here's how it works:

| Technique | What it does |
|---|---|
| **Quantization** | Compresses model weights from 16-bit to 4-bit (75% less memory) |
| **LoRA** | Only trains a tiny fraction of parameters (~0.44%) instead of all 1.2 billion |
| **Double Quantization** | Quantizes the quantization constants themselves for extra savings |
| **NF4 Format** | Special 4-bit format designed specifically for neural network weights |

The result — we fine-tuned a 1.2 billion parameter model on a **free Kaggle T4 GPU (15 GB VRAM)** with no quality loss compared to expensive full fine-tuning.

> **Entire project trained on free Kaggle GPUs and Google Colab. Zero cost.**

---

## The Dataset — `argilla/dpo-mix-7k`

Most fine-tuning datasets are huge and noisy. This one is different.

[Argilla](https://huggingface.co/datasets/argilla/dpo-mix-7k) created a carefully curated "cocktail" of exactly **7,500 high-quality conversational pairs**, mixing three distinct datasets in equal proportions:

| Source | Share | What it teaches |
|---|---|---|
| **Capybara** | 33% | Multi-turn conversations — how to hold a natural back-and-forth chat without losing context |
| **Intel Orca** | 33% | Step-by-step reasoning — how to break down complex instructions logically |
| **UltraFeedback** | 33% | Diverse prompts graded by GPT-4 on helpfulness, honesty, and harmlessness |

Each example in the dataset has two responses:
- **Chosen** — the better, more helpful response
- **Rejected** — a worse, less helpful response

We used the **chosen responses** for SFT (teaching format) and **both chosen + rejected** for DPO (teaching preference).

---

## The Training Pipeline

### Stage 1 — Supervised Fine-Tuning (SFT)

**Goal:** Teach the model the format and style of good responses.

We extract the `chosen` column from the dataset and train the model to produce those high-quality responses. Think of this as showing the model thousands of "this is what a great answer looks like" examples.

**Key settings:**
- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Quantization: 4-bit NF4 (QLoRA)
- LoRA rank: 16, alpha: 32
- Learning rate: 5e-5 with cosine scheduler
- Gradient clipping: 0.3 (prevents loss explosion)
- Steps: 300

**SFT Training Charts:**

![SFT Training](https://github.com/pranav6905/llm-finetuning-pipeline/blob/main/assests/sft_training.png)

What the charts show:
- **train/loss** — dropped from 1.9 → 1.5 and stabilized. The model is learning.
- **train/grad_norm** — fell from 0.04 → 0.01 and stayed flat. No gradient explosion, training is stable.
- **train/learning_rate** — cosine decay curve working correctly, rate reduces smoothly.
- **train/mean_token_accuracy** — climbed from 0.58 → 0.64, model predicts correct tokens more often.

---

### Stage 2 — DPO Alignment (Direct Preference Optimization)

**What is RLHF and why DPO?**

After SFT, the model knows how to respond in the right format. But it doesn't know which of two responses is *better*. This is where alignment comes in.

**RLHF (Reinforcement Learning from Human Feedback)** is the umbrella technique that teaches models to align with human preferences. It has two main approaches:

- **PPO (Proximal Policy Optimization)** — the classic approach used by ChatGPT. Requires training a separate reward model, very complex, needs lots of memory.
- **DPO (Direct Preference Optimization)** — a newer, simpler approach that achieves the same result without a reward model. Mathematically equivalent to PPO but far more stable and memory-efficient.

We chose **DPO** because it's more practical for limited hardware and produces equally good alignment results.

**How DPO works:**

```
For each prompt, the dataset has:
  Chosen response  ← the better answer
  Rejected response ← the worse answer

DPO trains the model to:
  Increase probability of chosen ↑
  Decrease probability of rejected ↓
  Maximize the gap between them (margin) ↑
```

**DPO Training Charts:**

![DPO Training](https://github.com/pranav6905/llm-finetuning-pipeline/blob/main/assests/dpo_training.png)

What the charts show:
- **rewards/chosen** — increasing -> Model assigns higher scores to good responses
- **rewards/rejected** — decreasing -> Model assigns lower scores to bad responses
- **rewards/margins** — increasing -> The gap between good and bad is growing
- **rewards/accuracies** — trending up from 0.30 → 0.55 -> Model correctly identifies the better response more often

All four metrics moving in the right direction confirms **DPO worked correctly.**

---

## Evaluation Results

Evaluated using [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) on two standard benchmarks:

| Model | ARC-Easy | TruthfulQA | Notes |
|---|---|---|---|
| Base LLaMA 3.2-1B | 59.81% | 25.21% | Pretrained weights, no fine-tuning |
| + SFT | 63.26% | 26.32% | After Stage 1 |
| **+ DPO** | **62.88%** | **26.93%** | After Stage 2 |

**About the -0.38% ARC drop after DPO:**
This is completely expected and has a name — the **alignment tax**. When you align a model to prefer honest, safe responses, it sometimes trades a tiny amount of raw task accuracy for better judgment. The TruthfulQA improvement (+0.61%) confirms DPO is doing exactly what it should — making the model more truthful.

---

## Deployment

The model is deployed as a live web application using a **FastAPI + Streamlit + Docker** stack on HuggingFace Spaces.

```
User (Browser)
     ↓
Streamlit Frontend (port 7860)
     ↓ HTTP request
FastAPI Backend (port 8000)
     ↓
LLaMA Model (generates response)
     ↓ JSON response
Streamlit displays result
```

**Why this stack?**

- **FastAPI** — wraps the model as a proper REST API with a `/generate` endpoint. This means the model can be called from any application, not just the UI.
- **Streamlit** — provides a clean, interactive UI that anyone can use without writing code.
- **Docker** — packages everything (Python version, dependencies, both services) into one container. Eliminates "works on my machine" problems completely.

**API endpoint:**
```bash
POST /generate
{
  "instruction": "Explain what inflation is",
  "max_new_tokens": 200,
  "temperature": 0.7
}
```

**Auto-generated API docs available at `/docs` (FastAPI's built-in Swagger UI).**

---

## Project Structure

```
llm-finetuning-pipeline/
│
├── notebooks/
│   ├── 01_sft_training.ipynb      # Stage 1: QLoRA + SFT training
│   ├── 02_dpo_training.ipynb      # Stage 2: DPO alignment
│   ├── 03_evaluation.ipynb        # lm-eval-harness benchmarks
│   └── 04_merge_export.ipynb      # Merge LoRA adapters and export full fine-tuned model to HuggingFace
│
├── app/
│   ├── main.py                    # FastAPI backend
│   └── frontend.py                # Streamlit UI
│
├── assets/
│   ├── sft_training.png           # SFT WandB charts
│   └── dpo_training.png           # DPO WandB charts
│
├── Dockerfile                     # Container definition
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Base Model | `meta-llama/Llama-3.2-1B-Instruct` |
| Fine-tuning | HuggingFace `transformers` + `peft` |
| Quantization | `bitsandbytes` 4-bit NF4 (QLoRA) |
| SFT Training | `trl` SFTTrainer |
| DPO Alignment | `trl` DPOTrainer |
| Experiment Tracking | Weights & Biases |
| Evaluation | `lm-evaluation-harness` |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker |
| Deployment | HuggingFace Spaces (free tier) |
| Training Hardware | Kaggle T4 GPU (free) + Google Colab |

---

## Future Work

### MT-Bench Evaluation
MT-Bench is a more comprehensive benchmark that tests multi-turn conversation quality using GPT-4 as the judge. It would give a much richer picture of model quality than ARC-Easy alone. Unfortunately it requires an OpenAI API key which has a cost, so it wasn't included here.

### vLLM Inference
[vLLM](https://github.com/vllm-project/vllm) is a production inference engine that makes LLM responses significantly faster through a technique called **PagedAttention** — it manages GPU memory for the KV cache the same way an OS manages virtual memory, eliminating waste and allowing much higher throughput. The current deployment uses standard HuggingFace inference on CPU which is slow (~30-60 seconds per response). vLLM would reduce this to 1-2 seconds. However, vLLM requires a dedicated GPU server at runtime, and no free tier service provides this. This would be the first upgrade in a production setting.

### Larger Models
The same pipeline works on larger models (7B, 13B). The only limitation is GPU memory — a 7B model needs at least 8-10 GB VRAM even in 4-bit, which requires a paid Colab or cloud GPU.

### Task-Specific Adapters
One of the best use cases for LoRA adapters is building a library of them — one adapter for coding, one for medical Q&A, one for legal documents — all sitting on top of the same base model. This is how companies like Microsoft deploy fine-tuned models efficiently in production.

---
## Quick Start
### Option 1: Docker (Recommended)
The project is containerized to run both the FastAPI backend and Streamlit frontend simultaneously.

```
# Clone and enter the repo
git clone https://github.com/pranav6905/llm-finetuning-pipeline.git
cd llm-finetuning-pipeline

# Build and run with Docker
docker build -t llm-pipeline .
docker run -p 7860:7860 -p 8000:8000 llm-pipeline
```
UI: http://localhost:7860
API Docs: http://localhost:8000/docs

### Option 2: Local Installation
```
# Install dependencies
pip install -r requirements.txt

# Run Backend (Terminal 1)
uvicorn app.main:app --port 8000

# Run Frontend (Terminal 2)
streamlit run app/frontend.py --server.port 7860
```
---

## Author

**Pranav** — CS Student passionate about ML/AI
- GitHub: [@pranav6905](https://github.com/pranav6905)
- HuggingFace: [@pranav6905](https://huggingface.co/pranav6905)

---

*Built with 🧠 and free GPUs.*
