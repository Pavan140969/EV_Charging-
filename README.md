---
title: EV-Grid-Optimizer
emoji: ⚡
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# ⚡ EV-GRID OPTIMIZER (OpenEnv Hackathon)

A high-fidelity **Multi-Agent Reinforcement Learning (RL) Environment** designed to solve the "Grid Stability Challenge" in modern urban EV infrastructures.

---

## 🎯 The Core Problem
As EV adoption grows, simultaneous charging creates massive "Grid Spikes." Existing systems are reactive. **EV-Grid Optimizer** is a proactive simulation environment where RL agents learn to:
1.  **Prioritize Emergency Vehicles**: Zero-latency response for critical transport.
2.  **Maximize Green Energy**: Shift high-drain charging to solar-rich windows.
3.  **Balance Load**: Prevent grid brownouts by intelligently distributing power across 10 stations.

## 🧠 Reinforcement Learning Design (The "OpenEnv" Way)
This project follows the **OpenEnv API** (`reset()`, `step()`) making it ready for immediate training with models like Qwen or Llama via TRL.

### Reward Function Components:
- ✅ **Base Uptime (+1.0)**: Encourages keeping stations operational.
- 🟢 **Solar Bonus (+1.0x)**: Rewards agents for using renewable energy.
- 🟡 **Queue Penalty (-0.1/car)**: Penalties for customer wait times.
- ❌ **Emergency Penalty (-3.0)**: High penalty for failing to prioritize emergency vehicles.

## 🚀 Key Features for Judges
- **Production-Ready Dashboard**: Real-time WebSocket updates with glassmorphism UI.
- **Isolated Sessions**: Each user/agent gets a private environment instance (scaling to 100+ concurrent users).
- **Dockerized & Deployable**: Built for Hugging Face Spaces free tier.
- **Optimized Simulation**: Asynchronous simulation engine for low-latency RL rollouts.

---

## 🛠️ Technical Stack
- **Backend**: FastAPI (Python)
- **Real-time**: WebSockets (AsyncIO)
- **UI**: TailwindCSS (Modern Neon Theme)
- **Environment**: OpenEnv Wrapper Compliance

## 📦 Setup & Deployment
1. **Local**: `uvicorn server.app:app --host 127.0.0.1 --port 8001`
2. **Container**: `docker build -t ev-optimizer .`
3. **HF Spaces**: Push to repo with `MAX_CONCURRENT_ENVS=100`.
