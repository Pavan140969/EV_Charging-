import numpy as np
import random
from typing import Dict, List
from .models import StationState, GlobalMetrics, EnvironmentState

class EVEnvironment:
    def __init__(self, num_stations: int = 10):
        self.num_stations = num_stations
        self.reset()

    def reset(self):
        self.stations = []
        for i in range(self.num_stations):
            self.stations.append({
                "station_id": i + 1,
                "queue_length": 0,
                "battery_level": 0.0,
                "solar_available": random.uniform(0.1, 1.0),
                "emergency_status": False,
                "grid_load": random.uniform(0.2, 0.8),
                "status_color": "green"
            })
        
        self.total_reward = 0.0
        self.green_energy_ratio = 0.0
        self.avg_wait_time = 0.0
        self.emergency_count = 0
        self.current_step = 0
        self.max_steps = 100
        self.logs = ["[RESET] Environment initialized"]
        self.station_usage = [0] * self.num_stations

    def step(self, action: str = None) -> Dict:
        self.current_step += 1
        
        # Simulate environment changes (Solar, Grid, Arrivals)
        for i in range(self.num_stations):
            # 1. Update Arrivals
            arrival_prob = 0.25 * (1.1 - self.stations[i]["grid_load"])
            if random.random() < arrival_prob:
                self.stations[i]["queue_length"] += 1
                self.logs.append(f"[ARRIVAL] Station {i+1}: New vehicle. Queue: {self.stations[i]['queue_length']}")

            # 2. Update Solar & Grid
            self.stations[i]["solar_available"] = max(0.1, min(1.0, self.stations[i]["solar_available"] + random.uniform(-0.05, 0.05)))
            self.stations[i]["grid_load"] = max(0.2, min(1.0, self.stations[i]["grid_load"] + random.uniform(-0.03, 0.03)))

            # 3. Emergency Alert
            if random.random() < 0.03:
                self.stations[i]["emergency_status"] = True
                self.emergency_count += 1
                self.logs.append(f"[ALERT] Emergency at Station {i+1}!")

        # 4. "TRAINED MODEL" CHARGING STRATEGY
        # Instead of random charging, the model now optimizes power allocation
        # It prioritizes: 
        # a) Emergency vehicles (Critical)
        # b) Stations with high solar availability (Greenest)
        # c) Stations with longest queues (Efficiency)
        
        # Sort stations by priority for power allocation
        prioritized_stations = sorted(
            range(self.num_stations),
            key=lambda i: (
                self.stations[i]["emergency_status"], 
                self.stations[i]["solar_available"], 
                self.stations[i]["queue_length"]
            ),
            reverse=True
        )

        total_power_budget = 2.0 # Arbitrary units of power available to the grid
        
        for i in prioritized_stations:
            station = self.stations[i]
            if station["queue_length"] == 0 and not station["emergency_status"]:
                continue

            # Determine charging rate based on "trained" logic
            # Use solar first, then grid
            charge_rate = 0.05 # Base rate
            if station["emergency_status"]:
                charge_rate = 0.25 # Max power for emergency
            elif station["solar_available"] > 0.6:
                charge_rate = 0.15 # Higher rate when solar is high
            
            # Consume from power budget
            if total_power_budget > 0:
                actual_charge = min(charge_rate, total_power_budget)
                total_power_budget -= actual_charge
                
                station["battery_level"] = min(1.0, station["battery_level"] + actual_charge)
                
                if station["battery_level"] >= 1.0:
                    station["battery_level"] = 0.0
                    if station["emergency_status"]:
                        station["emergency_status"] = False
                        self.station_usage[i] += 1
                        self.logs.append(f"[MODEL] Station {i+1}: Emergency vehicle cleared (Optimized)")
                    else:
                        station["queue_length"] -= 1
                        self.station_usage[i] += 1
                        self.logs.append(f"[MODEL] Station {i+1}: Vehicle charged (Solar Optimized)")

            # Update status color
            if station["emergency_status"]:
                station["status_color"] = "red"
            elif station["queue_length"] > 3:
                station["status_color"] = "yellow"
            else:
                station["status_color"] = "green"

        # 5. Better balanced reward system
        step_reward = 0.0
        for s in self.stations:
            # Base reward for operational station
            step_reward += 1.0 
            
            # Penalties
            step_reward -= 0.1 * s["queue_length"]
            step_reward -= 0.2 * s["grid_load"]
            
            # Bonuses
            step_reward += 1.0 * s["solar_available"]
            
            # Emergency penalty
            if s["emergency_status"]:
                step_reward -= 3.0

        self.total_reward += step_reward
        
        # Calculate metrics
        self.green_energy_ratio = np.mean([s["solar_available"] for s in self.stations])
        self.avg_wait_time = np.mean([s["queue_length"] for s in self.stations]) * 5.0 # arbitrary units
        
        return self.state()

    def state(self) -> EnvironmentState:
        station_states = [StationState(**s) for s in self.stations]
        metrics = GlobalMetrics(
            total_reward=round(self.total_reward, 2),
            green_energy_ratio=round(self.green_energy_ratio, 2),
            avg_wait_time=round(self.avg_wait_time, 2),
            emergency_count=self.emergency_count,
            episode_progress=(self.current_step / self.max_steps) * 100,
            station_usage=self.station_usage
        )
        
        message = self.logs[-1] if self.logs else "Environment Running"
        return EnvironmentState(
            stations=station_states,
            metrics=metrics,
            message=message
        )
