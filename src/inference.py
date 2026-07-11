import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

HF_API_URL = os.getenv("HF_API_URL", "https://dexq42rvmzmwby14.us-east-1.aws.endpoints.huggingface.cloud")
HF_TOKEN = os.getenv("HF_TOKEN")

# Prompt template mapping SFT/DPO formats
prompt_format = """Below is an instruction that describes a customer support scenario for practiceyourspeech.com. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

def generate_answer(question: str) -> str:
    """Calls the Hugging Face Inference Endpoint API using vLLM's OpenAI-compatible completions route."""
    if not HF_API_URL:
        return "Error: HF_API_URL environment variable is not set."
        
    formatted_prompt = prompt_format.format(instruction=question)
    
    # Format URL to target vLLM's /v1/completions route
    base_url = HF_API_URL.strip().rstrip("/")
    if not base_url.endswith("/v1/completions") and not base_url.endswith("/v1/chat/completions"):
        post_url = f"{base_url}/v1/completions"
    else:
        post_url = base_url
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}" if HF_TOKEN else "",
        "Content-Type": "application/json"
    }
    
    payload = {
        "prompt": formatted_prompt,
        "max_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(post_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Parse OpenAI-compatible vLLM text completion response
        if "choices" in result and len(result["choices"]) > 0:
            generated_text = result["choices"][0].get("text", "").strip()
        else:
            # Fallback in case of raw Hugging Face/TGI response format
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "").strip()
            elif isinstance(result, dict):
                generated_text = result.get("generated_text", "").strip()
            else:
                generated_text = str(result)
            
        # Clean up prompt prefix if it's returned by the container
        if "### Response:" in generated_text:
            generated_text = generated_text.split("### Response:")[-1].strip()
        return generated_text
        
    except Exception as e:
        return f"Error calling Hugging Face Endpoint API: {e}"

if __name__ == "__main__":
    # Example execution per assignment criteria
    question = "Why is the Subscribe button greyed out on my Profile page?"
    print(f"\n❓ Question: {question}")
    print("🔄 Sending request to Hugging Face Inference Endpoint...")
    answer = generate_answer(question)
    print(f"💡 Answer:\n{answer}\n")
    
    # Interactive mode option
    print("-" * 50)
    print("Interactive Mode. Type 'exit' to quit.")
    while True:
        user_q = input("\nAsk Support Agent: ")
        if user_q.lower().strip() == 'exit':
            break
        if not user_q.strip():
            continue
        ans = generate_answer(user_q)
        print(f"Agent Response:\n{ans}")

