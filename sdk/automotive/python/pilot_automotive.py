"""
Pilot example for automotive using QuNimbus SDK
"""

from qunimbus import QuNimbusClient


def main():
    client = QuNimbusClient()
    workload = {
        "type": "qpe",
        "vertical": "automotive",
        "parameters": {"precision": "high", "optimization": "enabled"},
    }
    job_id = client.submit_job(workload)
    results = client.get_results(job_id)
    print(f"automotive pilot: {results}")


if __name__ == "__main__":
    main()
