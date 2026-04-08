import requests
import json
import time
import os

# Base URL for the local environment
BASE_URL = os.getenv("BASE_URL", "http://localhost:7860")

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
        action = {"type": "step"}
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
