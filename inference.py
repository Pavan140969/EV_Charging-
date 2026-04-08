import requests
import json
import time
import os
from openai import OpenAI

# Base URL for the local environment
BASE_URL = os.getenv("BASE_URL", "http://localhost:7860")

# LLM Proxy configuration from environment variables
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")

# Initialize OpenAI client only if proxy is configured
client = None
if API_BASE_URL and API_KEY:
    print(f"Initializing LLM client with proxy: {API_BASE_URL}")
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
else:
    print("Warning: API_BASE_URL or API_KEY not set. LLM calls will be bypassed.")

def get_llm_action(observation):
    """Call the LLM proxy to determine the next action based on observation."""
    if not client:
        return {"type": "step"} # Fallback if no LLM configured
        
    try:
        # Construct prompt for the LLM
        prompt = f"Current environment state: {json.dumps(observation)}\n\n"
        prompt += "Choose the next action: 'step', 'reset', or 'run'. Reply with a JSON object like {'type': 'step'}."
        
        response = client.chat.completions.create(
            model="gpt-4o", # Or any model available on the proxy
            messages=[
                {"role": "system", "content": "You are an EV charging optimizer agent."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )
        
        llm_response = response.choices[0].message.content
        return json.loads(llm_response)
    except Exception as e:
        print(f"LLM call failed: {e}")
        return {"type": "step"}

def run_inference():
    print(f"Starting OpenEnv Inference at {BASE_URL}...")
    
    # Wait for server to be ready
    for _ in range(10):
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                break
        except:
            pass
        print("Waiting for server...")
        time.sleep(2)

    # 1. Reset the environment
    print("Resetting environment...")
    # Try both /reset and /openenv/reset
    try:
        response = requests.post(f"{BASE_URL}/openenv/reset")
        if response.status_code == 404:
            response = requests.post(f"{BASE_URL}/reset")
    except Exception as e:
        print(f"Connection error: {e}")
        return
        
    if response.status_code != 200:
        print(f"Error resetting environment: {response.status_code} {response.text}")
        return
    
    result = response.json()
    obs = result.get("observation")
    print("Environment Reset successfully.", flush=True)
    
    # REQUIRED: Print START block
    print("[START] task=EV-Grid-Optimizer", flush=True)
    
    total_reward = 0.0
    num_steps = 10 # Increased to 10 for better evaluation
    
    # 2. Run a few steps
    for i in range(num_steps):
        # Get action from LLM proxy
        action = get_llm_action(obs)
        
        try:
            response = requests.post(f"{BASE_URL}/openenv/step", json=action)
            if response.status_code == 404:
                response = requests.post(f"{BASE_URL}/step", json=action)
        except Exception as e:
            print(f"Step {i+1} connection error: {e}", flush=True)
            break
            
        if response.status_code != 200:
            print(f"Error in step {i+1}: {response.text}", flush=True)
            break
            
        result = response.json()
        obs = result.get("observation")
        reward = result.get("reward", 0.0)
        total_reward += reward
        
        # REQUIRED: Print STEP block
        print(f"[STEP] step={i+1} reward={reward}", flush=True)
        
        time.sleep(0.1)
    
    # REQUIRED: Print END block
    # score can be average reward or total reward based on requirement, usually total or normalized.
    # Let's use total_reward for score.
    print(f"[END] task=EV-Grid-Optimizer score={total_reward} steps={num_steps}", flush=True)
    
    print("Inference completed successfully.", flush=True)

if __name__ == "__main__":
    run_inference()
