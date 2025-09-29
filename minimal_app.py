from fastapi import FastAPI
from datetime import datetime

app = FastAPI(title="Minimal Test App")


@app.get("/ping")
async def ping():
    return {"pong": True, "time": datetime.utcnow().isoformat()}


@app.on_event("startup")
async def startup():
    print("[MINIMAL] startup event")


@app.on_event("shutdown")
async def shutdown():
    print("[MINIMAL] shutdown event")
