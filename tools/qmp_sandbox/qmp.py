import json
import os
import random
import sys
import time

BASE = float(os.getenv("QMP_BASE_PRICE_USD", "0.0004"))
ETA = 0.93


def price_stream(n=50, latency_max=60):
    start = time.time()
    books = []
    for i in range(n):
        eff = ETA + random.uniform(-0.02, 0.02)
        p = BASE * eff
        books.append({"ts": time.time(), "eff": eff, "price": p})
        time.sleep(random.uniform(0, latency_max / n))  # Simulate latency
    elapsed = time.time() - start
    return books, float(elapsed)


if __name__ == "__main__":
    try:
        books, lat = price_stream()
        res = {"stream": books, "total_latency_s": lat}
        print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
