import uvicorn, os
from app.main import app

print("[DEBUG_SCRIPT] Starting explicit uvicorn.run()")
uvicorn.run(app, host="127.0.0.1", port=8040, log_level="debug")
print("[DEBUG_SCRIPT] uvicorn.run() returned (server stopped)")
