from typing import Dict, Any, Tuple
from server.ev_environment import EVEnvironment


class OpenEnv:
    def __init__(self):
        self.env = EVEnvironment()
        self.done = False

    def reset(self) -> Dict[str, Any]:
        self.env.reset()
        self.done = False
        return self._observation()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.done:
            return self._observation(), 0.0, True, {"message": "episode finished"}
        a_type = action.get("type", "step")
        if a_type == "reset":
            self.env.reset()
        else:
            self.env.step(a_type)
        obs = self._observation()
        reward = obs["metrics"]["total_reward"]
        self.done = obs["metrics"]["episode_progress"] >= 100.0
        info = {"message": obs["message"]}
        return obs, reward, self.done, info

    def _observation(self) -> Dict[str, Any]:
        state = self.env.state().model_dump()
        return state

OpenEnvWrapper = OpenEnv
