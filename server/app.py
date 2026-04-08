import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from .ev_environment import EVEnvironment
from .models import Action
import os

app = FastAPI()

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

MAX_CONCURRENT_ENVS = int(os.getenv("MAX_CONCURRENT_ENVS", "100"))
sessions = {}

@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/web")
async def get_web():
    return FileResponse(os.path.join(static_dir, "index.html"))

async def send_state(ws: WebSocket, env: EVEnvironment):
    state = env.state().model_dump_json()
    await ws.send_text(state)

async def run_episode(ws: WebSocket, env: EVEnvironment, speed: float = 1.0):
    try:
        while env.current_step < env.max_steps:
            env.step()
            await send_state(ws, env)
            await asyncio.sleep(1.0 / speed)
    except Exception:
        pass

async def heartbeat(ws: WebSocket):
    """Keep the connection alive by sending a ping every 20 seconds."""
    try:
        while True:
            await asyncio.sleep(20)
            await ws.send_json({"type": "ping"})
    except Exception:
        pass

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if len(sessions) >= MAX_CONCURRENT_ENVS:
        await websocket.close(code=1013)
        return
    env = EVEnvironment()
    task = None
    hb_task = asyncio.create_task(heartbeat(websocket))
    sessions[websocket] = {"env": env, "task": task, "hb_task": hb_task}
    await send_state(websocket, env)
    try:
        while True:
            data = await websocket.receive_text()
            action_data = json.loads(data)
            
            # Handle pong from client
            if action_data.get("type") == "pong":
                continue
                
            action = Action(**action_data)
            current = sessions.get(websocket)
            if not current:
                break
            if current["task"] and not current["task"].done():
                current["task"].cancel()
            if action.type == "run":
                current["task"] = asyncio.create_task(run_episode(websocket, env, action.speed or 1.0))
            elif action.type == "step":
                env.step()
                await send_state(websocket, env)
            elif action.type == "reset":
                env.reset()
                await send_state(websocket, env)
            sessions[websocket] = current
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        current = sessions.pop(websocket, None)
        if current:
            if current.get("task") and not current["task"].done():
                current["task"].cancel()
            if current.get("hb_task") and not current["hb_task"].done():
                current["hb_task"].cancel()
        try:
            await websocket.close()
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host=host, port=port)
