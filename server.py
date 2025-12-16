import hashlib
import time

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/qd/health", methods=["GET"])
def qd_health():
    t0 = time.perf_counter()
    resp = {"status": "OK"}
    resp["_t_ms"] = int((time.perf_counter() - t0) * 1000)
    return jsonify(resp)


@app.route("/api/health", methods=["GET"])
def api_health():
    t0 = time.perf_counter()
    resp = {"status": "OK"}
    resp["_t_ms"] = int((time.perf_counter() - t0) * 1000)
    return jsonify(resp)


@app.route("/explain", methods=["POST"])
def explain():
    """
    POST JSON:
    {
      "commit": "",
      "simulate_ms": 0   # optional: simulate LLM latency in ms
    }
    """
    data = request.get_json()
    if data is None:
        data = {}
    text = data.get("commit", "")
    simulate_ms = int(data.get("simulate_ms", 0))

    t0 = time.perf_counter()

    # Simulated "work":
    # - compute sha256 (simulates object hashing)
    # - optionally sleep to simulate LLM latency
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()

    if simulate_ms > 0:
        time.sleep(simulate_ms / 1000.0)

    # Simple deterministic "explanation": first sentence + SHA snippet
    first_sentence = text.split(".")[0].strip() if text else "No commit message provided."
    explanation = f"Explanation: {first_sentence} (sha:{sha[:8]})"

    t_ms = int((time.perf_counter() - t0) * 1000)
    return jsonify({"explanation": explanation, "processing_ms": t_ms})


if __name__ == "__main__":
    # Flask development server is threaded by default (since Flask 2.0+)
    app.run(host="0.0.0.0", port=5000)
