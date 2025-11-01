# Observability Overview

* **Metrics**: Exposes Prometheus-compatible metrics at `/metrics` using `prometheus_client`. Configure Prometheus with the provided scrape config.
* **Logging**: Structured JSON logging to stdout; aggregate via CloudWatch Logs or Loki.
* **Tracing**: Tempo or AWS X-Ray integration achievable through OpenTelemetry exporters configured in Gunicorn workers.
* **Dashboards**: Import the provided Grafana dashboard JSON to visualise kernel latency, throughput, and error rates.
