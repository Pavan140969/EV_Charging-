from pydantic import BaseModel
from typing import List, Optional

class StationState(BaseModel):
    station_id: int
    queue_length: int
    battery_level: float
    solar_available: float
    emergency_status: bool
    grid_load: float
    status_color: str

class GlobalMetrics(BaseModel):
    total_reward: float
    green_energy_ratio: float
    avg_wait_time: float
    emergency_count: int
    episode_progress: float
    station_usage: List[int] # Vehicles charged per station

class EnvironmentState(BaseModel):
    stations: List[StationState]
    metrics: GlobalMetrics
    message: str

class Action(BaseModel):
    type: str  # 'run', 'step', 'reset'
    speed: Optional[float] = 1.0
