import time, datetime

print("Sleep test starting...")
try:
    while True:
        print("Heartbeat", datetime.datetime.utcnow().isoformat(), flush=True)
        time.sleep(5)
except KeyboardInterrupt:
    print("Interrupted")
