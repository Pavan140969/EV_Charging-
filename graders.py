import numpy as np

def grade_emergency_priority(observation, info):
    """Grades how well emergency vehicles were prioritized."""
    # Score is high if emergency_count is low or successfully cleared
    metrics = observation.get("metrics", {})
    emergency_count = metrics.get("emergency_count", 0)
    
    # Simple normalization: score between 0.1 and 0.9
    # Fewer emergencies remaining = better score
    score = max(0.1, min(0.9, 1.0 - (emergency_count * 0.1)))
    return score

def grade_green_energy_usage(observation, info):
    """Grades the efficiency of solar energy usage."""
    metrics = observation.get("metrics", {})
    ratio = metrics.get("green_energy_ratio", 0.5)
    
    # Keep strictly between 0 and 1
    score = max(0.1, min(0.9, ratio))
    return score

def grade_queue_efficiency(observation, info):
    """Grades the average wait time / queue length."""
    metrics = observation.get("metrics", {})
    avg_wait = metrics.get("avg_wait_time", 0.0)
    
    # Lower wait time = higher score
    # Assume 50 is a 'bad' wait time for normalization
    score = max(0.1, min(0.9, 1.0 - (avg_wait / 50.0)))
    return score

# Task definitions for OpenEnv
TASKS = [
    {
        "name": "emergency_priority",
        "grader": grade_emergency_priority,
        "description": "Ensure emergency vehicles are handled with top priority."
    },
    {
        "name": "green_energy",
        "grader": grade_green_energy_usage,
        "description": "Maximize the usage of solar power at charging stations."
    },
    {
        "name": "queue_optimization",
        "grader": grade_queue_efficiency,
        "description": "Minimize average vehicle wait times across the grid."
    }
]
