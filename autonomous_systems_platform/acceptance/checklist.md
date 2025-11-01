# Acceptance Checklist

- [ ] Terraform plan executed without errors.
- [ ] Kubernetes namespace `autonomous-systems` created.
- [ ] `/health` endpoint returns 200.
- [ ] `/kernel` endpoint returns deterministic JSON for seed 0.
- [ ] `/metrics` exposes Prometheus data including request counters.
- [ ] Frontend dashboard displays latest kernel output.
- [ ] CI pipeline succeeded (build, test, deploy).
