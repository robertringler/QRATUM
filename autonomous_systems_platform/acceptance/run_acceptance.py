import json
import os
from urllib import request

BASE_URL = os.getenv("ACCEPTANCE_BASE_URL", "http://localhost:8000")

def fetch(path: str):
    with request.urlopen(f"{BASE_URL}{path}") as resp:
        return resp.read().decode(), resp.status

def main():
    _, status = fetch("/health")
    assert status == 200
    kernel_req = request.Request(
        f"{BASE_URL}/kernel",
        data=json.dumps({"scale": 1.0}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(kernel_req) as resp:
        payload = json.loads(resp.read().decode())
        assert "result" in payload
    _, status = fetch("/metrics")
    assert status == 200
    print("Acceptance checks passed.")

if __name__ == "__main__":
    main()
