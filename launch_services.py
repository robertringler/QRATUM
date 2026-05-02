# Helper to launch backend and frontend reliably and capture logs
import os
import subprocess
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VENV_PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
LOG_DIR = os.path.join(ROOT, "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

backend_cmd = [
    VENV_PY,
    os.path.join(ROOT, "all_analyses.py"),
    "--Lx",
    "8",
    "--Ly",
    "8",
    "--disorder",
    "0.2",
    "--T",
    "5",
    "--dt",
    "0.02",
]
frontend_cmd = [
    VENV_PY,
    "-m",
    "streamlit",
    "run",
    os.path.join(ROOT, "quantum_simulator_dashboard", "app.py"),
    "--server.headless",
    "true",
]

backend_log = open(os.path.join(LOG_DIR, "backend_launch.log"), "w")
frontend_log = open(os.path.join(LOG_DIR, "frontend_launch.log"), "w")

print("Starting backend...")
backend_proc = subprocess.Popen(backend_cmd, stdout=backend_log, stderr=subprocess.STDOUT)
print("Backend PID:", backend_proc.pid)

print("Starting frontend...")
frontend_proc = subprocess.Popen(frontend_cmd, stdout=frontend_log, stderr=subprocess.STDOUT)
print("Frontend PID:", frontend_proc.pid)

# Wait a few seconds for frontend to start
time.sleep(4)

print("Done. Logs:")
print(" -", backend_log.name)
print(" -", frontend_log.name)

# Write PIDs to file for later termination
with open(os.path.join(LOG_DIR, "service_pids.txt"), "w") as f:
    f.write(f"backend:{backend_proc.pid}\n")
    f.write(f"frontend:{frontend_proc.pid}\n")

# Do not wait for processes; exit so the launching is detached
backend_log.close()
frontend_log.close()
print("Starter finished.")
