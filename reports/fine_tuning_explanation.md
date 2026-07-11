# Understanding LLM Fine-Tuning (LoRA, QLoRA, SFT, and DPO)

This document provides a conceptual and practical overview of the techniques, algorithms, and configurations used to build our domain-specific Customer Support Assistant using **Unsloth**.

---

## 1. Concept Explanations

### Why Full Fine-Tuning is Expensive
During full fine-tuning, every single weight and parameter of the Large Language Model is updated. For a model with billions of parameters, this requires:
- Huge memory to store optimizer states (e.g., AdamW stores first/second moments), gradients, and activations for every parameter.
- High compute power (multiple high-end GPUs like A100/H100).
- Risk of **Catastrophic Forgetting**, where the model updates its weights so much that it loses its general reasoning and logic capabilities.

### What LoRA (Low-Rank Adaptation) Does
LoRA addresses the cost of full fine-tuning by freezing the pretrained base model weights and injecting small, trainable parameter matrices into key attention layers. 
- Instead of updating a weight matrix $W$ of size $d \times d$, LoRA updates two low-rank matrices $A$ (size $d \times r$) and $B$ (size $r \times d$), where $r \ll d$ (typically $r = 8$ or $16$).
- The new weight is computed as $W + \Delta W$ where $\Delta W = B \times A$.
- This reduces the number of trainable parameters by **99%**, drastically cutting down GPU memory requirements while maintaining near-identical accuracy.

### What QLoRA (Quantized Low-Rank Adaptation) Does
QLoRA takes LoRA a step further by quantizing the base model weights to a highly efficient **4-bit NormalFloat (NF4)** format.
- It introduces Double Quantization (quantizing the quantization constants themselves to save extra memory).
- It utilizes a Paged Optimizer to manage GPU memory spikes by automatically page-swapping memory to CPU RAM during backward passes.
- Trainable LoRA matrices are kept in 16-bit float formats (Float16 or Bfloat16) to calculate gradients accurately.

### Why QLoRA is Useful on Limited GPU Memory
On a standard consumer GPU with limited memory (like the 16GB Tesla T4 in free Google Colab or Kaggle):
- Loading a standard 7B or 8B model in full precision would instantly crash the system due to Out of Memory (OOM) errors.
- QLoRA shrinks the base model's memory footprint by **75%**, allowing models up to 8B or 13B parameters to be loaded, trained, and tested on a single budget GPU.

### What is Non-Instruction Fine-Tuning?
Non-instruction fine-tuning (also called pretraining or domain adaptation) trains the model on raw, unstructured paragraphs of text. Using Causal Language Modeling (predicting the next token), the model internalizes vocabulary, spelling, technical structures, formulas, and background context before learning to act as an assistant.

### What is Instruction Fine-Tuning (SFT)?
Supervised Fine-Tuning (SFT) teaches the model how to follow instruction-based user requests. The training dataset consists of specific Prompt/Response or Question/Answer pairs, training the model to respond helpfully in a conversational format rather than just completing paragraphs.

### What is DPO (Direct Preference Optimization)?
DPO is a preference-alignment technique that replaces complex Reinforcement Learning from Human Feedback (RLHF). Instead of training a separate reward model and using PPO, DPO uses a classification loss function directly on pairs of preferred (chosen) and non-preferred (rejected) responses. This makes preference alignment stable, fast, and highly memory efficient.

### Difference between SFT and DPO
- **SFT** teaches the model *what to say* (imparts raw facts and basic Q&A behaviors).
- **DPO** teaches the model *how to say it* (aligns style, tone, safety, conciseness, and filters out hallucinations/technical leaks by choosing professional answers over poor ones).

---

## 2. Hyperparameter Settings Used

The following hyperparameters were selected for training our assistant in Unsloth:

- **LoRA Rank ($r$)**: `16`
  - *Reasoning*: A rank of 16 provides a balanced capacity to learn complex technical domain parameters without inflating the number of trainable weights.
- **LoRA Alpha ($\alpha$)**: `16`
  - *Reasoning*: Setting alpha equal to rank ($r=16$) ensures stable scaling of the adapter weights relative to the pretrained base weights.
- **LoRA Dropout**: `0.05` for pretraining, `0.0` for SFT/DPO
  - *Reasoning*: Slight dropout helps prevent overfitting during the raw text adaptation phase, while setting it to 0.0 is recommended for SFT/DPO to maximize convergence stability.
- **Learning Rate**: 
  - **SFT**: `2e-4` (Linear decay, optimized for fast and stable weight adjustments).
  - **DPO**: `5e-6` (Cosine decay, a much lower learning rate to make subtle adjustments to tone without degrading factual accuracy).
- **Batch Size Configuration**:
  - **Per Device Batch Size**: `4` for SFT, `2` for DPO.
  - **Gradient Accumulation Steps**: `4`.
  - **Effective Batch Size**: `16` for SFT, `8` for DPO.
  - *Reasoning*: Accumulating gradients allows us to simulate larger batch sizes on a single GPU without encountering OOM crashes.
