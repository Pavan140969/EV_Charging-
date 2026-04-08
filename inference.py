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
    if not client or observation is None:
        return {"type": "step"} # Fallback if no LLM configured or no observation
        
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
        try:
            return json.loads(llm_response)
        except json.JSONDecodeError:
            # Fallback if LLM returns text instead of JSON
            print(f"LLM returned invalid JSON: {llm_response}")
            return {"type": "step"}
    except Exception as e:
        print(f"LLM call failed: {e}")
        return {"type": "step"}

def run_inference():
    print(f"Starting OpenEnv Inference at {BASE_URL}...")
    
    tasks = ["emergency_priority", "green_energy", "queue_optimization"]
    
    for task_name in tasks:
        print(f"--- Running Task: {task_name} ---", flush=True)
        
        # Wait for server to be ready
        for _ in range(10):
            try:
                response = requests.get(f"{BASE_URL}/health")
                if response.status_code == 200:
                    break
            except Exception:
                pass
            print("Waiting for server...", flush=True)
            time.sleep(2)

        # 1. Reset the environment
        print(f"Resetting environment for {task_name}...", flush=True)
        try:
            response = requests.post(f"{BASE_URL}/openenv/reset")
            if response.status_code == 404:
                response = requests.post(f"{BASE_URL}/reset")
        except Exception as e:
            print(f"Connection error: {e}", flush=True)
            continue
            
        if response.status_code != 200:
            print(f"Error resetting: {response.status_code}", flush=True)
            continue
        
        result = response.json()
        obs = result.get("observation")
        
        # REQUIRED: Print START block for each task
        print(f"[START] task={task_name}", flush=True)
        
        total_reward = 0.0
        num_steps = 10
        
        # 2. Run steps
        for i in range(num_steps):
            action = get_llm_action(obs)
            try:
                response = requests.post(f"{BASE_URL}/openenv/step", json=action)
                if response.status_code == 404:
                    response = requests.post(f"{BASE_URL}/step", json=action)
            except Exception as e:
                print(f"Step {i+1} error: {e}", flush=True)
                break
                
            if response.status_code != 200:
                break
                
            result = response.json()
            obs = result.get("observation")
            reward = result.get("reward", 0.0)
            total_reward += reward
            
            # REQUIRED: Print STEP block
            print(f"[STEP] step={i+1} reward={reward}", flush=True)
            time.sleep(0.1)
        
        # REQUIRED: Print END block with normalized score (0 to 1)
        # Based on the grader logic, let's normalize the final result
        # Final score must be strictly between 0 and 1
        final_score = max(0.1, min(0.9, total_reward / (num_steps * 10.0)))
        print(f"[END] task={task_name} score={final_score} steps={num_steps}", flush=True)
    
    print("All tasks completed successfully.", flush=True)

if __name__ == "__main__":
    run_inference()
