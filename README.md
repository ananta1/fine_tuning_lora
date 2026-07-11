# Practice Your Speech - Domain-Specific Customer Support Assistant

An end-to-end domain-specific AI assistant built by fine-tuning an open-source LLM using **Unsloth** across three training stages: Non-Instruction Fine-Tuning, Supervised Fine-Tuning (SFT), and Direct Preference Optimization (DPO).

---

## 1. Domain Selection & Business Problem

*   **Selected Domain**: Customer Support Assistant / IT Helpdesk
*   **Target Platform**: `practiceyourspeech.com` (AI public speaking rehearsal SaaS)
*   **Business Problem**: Practice Your Speech utilizes an AWS serverless architecture combining AWS Transcribe, AWS Rekognition, and Amazon Bedrock. Due to complex backend workflows, subscription structures, and known codebase bugs (like Stripe payment float conversion failures or disabled subscription checkout buttons in the UI), standard general-purpose models fail to provide accurate customer troubleshooting. They yield generic, incorrect answers that fail to resolve the user's issues.
*   **Objective**: Train a domain-expert assistant that understands internal system architectures, billing/quota parameters, AWS integration details, and provides helpful, safe, and professional support responses.

---

## 2. Dataset Details

All dataset files are stored in the [data/](file:///c:/Users/bharg/2026/practiceyourspeech/domain-ai-assistant-finetuning/data/) directory:

1.  **Non-Instruction Dataset** ([non_instruction_data.txt](file:///c:/Users/bharg/2026/practiceyourspeech/domain-ai-assistant-finetuning/data/non_instruction_data.txt)):
    *   **Size**: 51 paragraphs of raw technical domain text.
    *   **Content**: Full descriptions of platform metrics (WPM, pause calculations, clarity indices), database tables (composite keys, GSIs), AWS services (S3 pre-signed URLs, Lambda Powertools), billing hooks, and troubleshooting helper scripts (`reprocess_stuck_video.py`).
2.  **Instruction Dataset** ([instruction_dataset.jsonl](file:///c:/Users/bharg/2026/practiceyourspeech/domain-ai-assistant-finetuning/data/instruction_dataset.jsonl)):
    *   **Size**: 105 instruction-response examples.
    *   **Content**: Conversational Q&A pairs covering account verification, password resets, payment bugs, quota resets, visual sentiment enums, stuck video reprocesses, and general speech guidelines.
3.  **Preference Dataset** ([preference_dataset.jsonl](file:///c:/Users/bharg/2026/practiceyourspeech/domain-ai-assistant-finetuning/data/preference_dataset.jsonl)):
    *   **Size**: 55 preference pairs (prompt, chosen, rejected).
    *   **Content**: Contrast pairs training the model to adopt a polite, customer-friendly support tone (chosen) rather than leaking raw code properties, internal database schema definitions, or providing generic responses (rejected).

---

## 3. Base Model Used

*   **Base Model**: `unsloth/Qwen2.5-7B` (Hugging Face)
*   **Format**: 4-bit quantized NormalFloat (NF4) wrapper provided by Unsloth.
*   **Rationale**: Upgraded from the 1.5B model to the larger 7B model to increase capacity, improve reasoning, and significantly reduce factual hallucinations. Still fits comfortably within a 16GB T4 GPU VRAM using Unsloth's 4-bit QLoRA optimizations.

---

## 4. Fine-Tuning Methodology

### LoRA / QLoRA Configuration
We applied Low-Rank Adaptation (LoRA) to all linear projection layers to maximize representational capacity:
```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0.05, # (0.05 for pretraining, 0.0 for SFT/DPO)
    bias = "none",
)
```

### Stage 1: Non-Instruction Fine-Tuning
*   **Objective**: Adapt the model's vocabulary and background knowledge to the platform's specific systems.
*   **Notebook**: [non_instruction_finetuning.ipynb](file:///c:/Users/bharg/2026/practiceyourspeech/domain-ai-assistant-finetuning/notebooks/non_instruction_finetuning.ipynb)
*   **Training Arguments**: Learning rate `2e-4`, batch size `4` with `4` accumulation steps (effective `16`), 60 max steps, linear decay.
*   **Observation**: Model adapted to the vocabulary (e.g. automatically auto-completing platform terms like WPM calculations).

### Stage 2: Supervised Fine-Tuning (SFT)
*   **Objective**: Train the model in structured Q&A behavior using prompt headers.
*   **Notebook**: [instruction_finetuning.ipynb](file:///c:/Users/bharg/2026/practiceyourspeech/domain-ai-assistant-finetuning/notebooks/instruction_finetuning.ipynb)
*   **Training Arguments**: Learning rate `2e-4`, batch size `4` with `4` accumulation steps, 100 max steps.
*   **Observation**: Model successfully answered technical questions but leaked developer terms (like database fields and variable names) directly to the user.

### Stage 3: Direct Preference Optimization (DPO)
*   **Objective**: Align tone, conciseness, and format using pairwise preference optimization.
*   **Notebook**: [dpo_alignment.ipynb](file:///c:/Users/bharg/2026/practiceyourspeech/domain-ai-assistant-finetuning/notebooks/dpo_alignment.ipynb)
*   **Training Arguments**: Learning rate `5e-6` (Cosine decay, low LR to avoid catastrophic drift), batch size `2` with `4` accumulation steps, beta `0.1` reward factor.
*   **Observation**: The final DPO model became highly professional, friendly, and brief, hiding backend code leaks while providing accurate help.

---

## 5. Before vs. After Output Comparison

| Question | Base Model Answer | SFT Model Answer | DPO Aligned Model Answer (Final) |
| :--- | :--- | :--- | :--- |
| **Why is the Subscribe button greyed out?** | "A greyed-out subscribe button usually indicates..." | "The subscription button on the Profile page is currently disabled (disabled={true}) and shows '$7/month' because the self-service checkout portal is undergoing maintenance..." | "The Subscribe button is currently disabled for maintenance. We apologize for the inconvenience. To upgrade immediately, please contact support and our team will manually activate your Pro plan." |
| **My payment succeeded but I haven't upgraded. Why?** | "Sometimes payments take time to process..." | "This is caused by a serialization error in our backend `/verify-payment` handler, which crashes when Stripe floats are written to DynamoDB without Decimal wrapping..." | "We're sorry for the upgrade delay. This occurs due to a transaction database sync error. Please contact us with your Stripe invoice reference and we will manually activate your Pro plan right away." |
| **My speech is stuck in 'IN_PROGRESS'. How do I fix it?** | "Processing times vary depending on network traffic..." | "Analysis usually takes 1-3 minutes. If it is stuck, an AWS job or Bedrock call may have failed. A support agent can run `python reprocess_stuck_video.py <userId> <videoKey>`..." | "Analysis typically takes 1-3 minutes. If it has been stuck for over 10 minutes, please contact support. Our team will check the status of your processing jobs and reset them if needed." |

---

## 6. Challenges Faced & Future Improvements

### Challenges
1.  **DynamoDB Serialization Awareness**: Training the model to recognize internal bugs without outputting python exceptions to users required precise DPO training.
2.  **Repetitive Output Loops**: SFT models easily slip into repetitive output phrasing when trained on a small instruction set. The low learning rate and cosine decay in the DPO phase successfully reduced this.

### Future Improvements
1.  **Multi-turn Chat History**: Expand SFT datasets to cover multi-turn troubleshooting conversations.
2.  **Context-Aware Diagnostics**: Integrate system telemetry into the prompt so the chatbot can check actual user database fields before responding.

---

## 7. Installation & Usage

1.  Clone the repository and install requirements:
    ```bash
    pip install -r requirements.txt
    ```
2.  Open notebooks in Google Colab to run the training workflows.
3.  Execute the final assistant inference script:
    ```bash
    python src/inference.py
    ```


## 8. Hugging Face Endpoint Deployment (vLLM)

To deploy the final merged model (`Bhargav1/qwen2.5-7b-final-merged`) using Hugging Face Inference Endpoints:

### Step-by-Step Deployment with vLLM
1. Delete any stuck/failed endpoints.
2. Click **Create new endpoint**.
3. Select your model repo: `Bhargav1/qwen2.5-7b-final-merged`.
4. In the **Task** dropdown, select **Text Generation**.
5. **Context Window Configuration (Important):**
   - Qwen2.5 models default to a massive 128K context window (`max_position_embeddings: 131072` in `config.json`). By default, vLLM will try to pre-allocate GPU memory for this entire context window, triggering a VRAM/CPU memory crash during container startup.
   - To bypass any container configuration environment variable bugs, edit the model's configuration file directly on Hugging Face Hub:
     1. Go to your model repository on Hugging Face: `https://huggingface.co/Bhargav1/qwen2.5-7b-final-merged`.
     2. Open **Files and versions** -> **`config.json`**.
     3. Click **Edit**, find `"max_position_embeddings": 131072`, and change it to `"max_position_embeddings": 4096` (or `2048`).
     4. Commit the changes directly to `main`.
6. In the **Advanced Configuration** section of the Endpoint setup, ensure the **Inference Engine** is set to **vLLM** (Hugging Face's recommended serving backend).
7. Choose the **Nvidia A10G** (or **L4**) GPU instance (24GB VRAM).
8. Click **Create Endpoint**.

### Why vLLM is Recommended
- **Direct GPU Loading:** Bypasses CPU memory bottlenecks, preventing OOM crashes during model initialization.
- **Auto-Precision:** Automatically loads the model in its native `bfloat16` precision (consuming ~14GB VRAM).
- **PagedAttention:** Optimizes VRAM allocation during runtime generation, delivering high token throughput.