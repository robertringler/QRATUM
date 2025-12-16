import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE = "http://127.0.0.1:5000"


def single_request(path, method="GET", json=None):
    url = BASE + path
    t0 = time.perf_counter()
    try:
        if method == "GET":
            r = requests.get(url, timeout=30)
        else:
            r = requests.post(url, json=json or {}, timeout=60)
        r.raise_for_status()  # Raise exception for 4xx/5xx status codes
        dt = (time.perf_counter() - t0) * 1000.0
        return dt, r.status_code, r.text[:200]
    except requests.exceptions.RequestException as e:
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"Request failed: {e}")
        return dt, 0, str(e)[:200]


def run_sequential(path, n=50, method="GET", json=None):
    latencies = []
    for _i in range(n):
        dt, status, _ = single_request(path, method=method, json=json)
        latencies.append(dt)
    return latencies


def run_concurrent(path, total=200, concurrency=20, method="GET", json=None):
    latencies = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(single_request, path, method, json) for _ in range(total)]
        for fut in as_completed(futures):
            dt, status, _ = fut.result()
            latencies.append(dt)
    return latencies


def summarize(latencies):
    lat_sorted = sorted(latencies)
    n = len(lat_sorted)
    return {
        "count": n,
        "mean_ms": statistics.mean(lat_sorted) if lat_sorted else None,
        "p50_ms": statistics.median(lat_sorted) if lat_sorted else None,
        "p90_ms": lat_sorted[min(int(n * 0.90), n - 1)] if n >= 10 else None,
        "p99_ms": lat_sorted[min(int(n * 0.99), n - 1)] if n >= 100 else None,
        "min_ms": min(lat_sorted) if lat_sorted else None,
        "max_ms": max(lat_sorted) if lat_sorted else None,
    }


if __name__ == "__main__":
    print("Running quick benchmarks against local server (ensure server.py is running)")

    # Health sequential
    seq = run_sequential("/qd/health", n=30, method="GET")
    print("QD /qd/health sequential:", summarize(seq))

    # API health concurrent
    conc = run_concurrent("/api/health", total=200, concurrency=40, method="GET")
    print("API /api/health concurrent:", summarize(conc))

    # Explain with no simulate latency (concurrent)
    conc_explain = run_concurrent(
        "/explain",
        total=100,
        concurrency=20,
        method="POST",
        json={"commit": "Fix bug in handler. Update tests."},
    )
    print("Explain (no simulate_ms) concurrent:", summarize(conc_explain))

    # Explain with simulated LLM latency (e.g., 300ms)
    conc_explain_sim = run_concurrent(
        "/explain",
        total=60,
        concurrency=10,
        method="POST",
        json={"commit": "Refactor module X for readability.", "simulate_ms": 300},
    )
    print("Explain (simulate 300ms) concurrent:", summarize(conc_explain_sim))
