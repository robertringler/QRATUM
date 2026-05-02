# Disconnected Frontend↔Backend seams
_Commit_: `8fc58a9107334a3b53b69a47580df64b185d3317`  _Generated_: 2026-04-29T22:27:58Z

Backend endpoints with **no** literal occurrence in any UI/dashboard/docs blob: **57** of 90 discovered.

> **Caveat**: this is a literal-string match. Routes built by string concatenation, URL templates, or proxied through reverse proxies will appear as DANGLING here even when wired.

| Kind | Backend file | Route/Event |
|---|---|---|
| route | `api/v1/jobs.py` | `/validate` |
| route | `api/v1/main.py` | `/readiness` |
| route | `api/v1/main.py` | `/v1/auth/token` |
| route | `api/v1/main.py` | `/v1/auth/refresh` |
| route | `api/v1/main.py` | `/v1/auth/revoke` |
| route | `api/v1/resources.py` | `/clusters` |
| route | `api/v1/resources.py` | `/quotas` |
| route | `api/v1/results.py` | `/jobs/{job_id}/results` |
| route | `api/v1/results.py` | `/jobs/{job_id}/artifacts` |
| route | `api/v1/results.py` | `/jobs/{job_id}/artifacts/{artifact_id}` |
| route | `api/v1/results.py` | `/jobs/{job_id}/visualization` |
| route | `api/v1/status.py` | `/jobs/{job_id}/status` |
| route | `docker/qratum/api_server.py` | `/api/v1/verticals` |
| route | `docker/qratum/api_server.py` | `/api/v1/vertical/{vertical_id}` |
| route | `docker/qratum/api_server.py` | `/api/v1/vertical/execute` |
| route | `docker/qratum/api_server.py` | `/api/v1/synthesis/execute` |
| route | `docker/qratum/api_server.py` | `/api/v1/synthesis/verify/{chain_id}` |
| route | `docker/qratum/api_server.py` | `/api/v1/stats` |
| route | `integrations/services/quasim-api/server.py` | `/readiness` |
| route | `integrations/services/quasim-api/server.py` | `/jobs/submit` |
| route | `integrations/services/quasim-api/server.py` | `/jobs/{job_id}/status` |
| route | `integrations/services/quasim-api/server.py` | `/artifacts/{artifact_id}` |
| route | `integrations/services/quasim-api/server.py` | `/profiles` |
| route | `integrations/services/quasim-api/server.py` | `/validate` |
| route | `qratum_ai_platform/services/model_server/app.py` | `/predict` |
| route | `qratum_ai_platform/services/orchestrator/app.py` | `/route` |
| route | `qratum_ai_platform/services/orchestrator/app.py` | `/routes` |
| route | `qratum_fullstack_server.py` | `/api/v1/qradle/contract` |
| route | `qratum_fullstack_server.py` | `/api/v1/qradle/contract` |
| route | `qratum_fullstack_server.py` | `/api/v1/platform/execute` |
| route | `qratum_fullstack_server.py` | `/api/v1/asi/status` |
| route | `qratum_fullstack_server.py` | `/api/v1/system/proof` |
| route | `qratum_fullstack_server.py` | `/api/v1/system/checkpoint` |
| route | `qratum_fullstack_server.py` | `/api/v1/audit/trail` |
| route | `qratum_fullstack_server.py` | `/api/v1/verticals` |
| route | `qratum_fullstack_server.py` | `/api/v1/soi/qradle/state` |
| route | `qratum_fullstack_server.py` | `/api/v1/soi/qradle/audit` |
| route | `qratum_fullstack_server.py` | `/api/v1/soi/qradle/checkpoints` |
| route | `qratum_fullstack_server.py` | `/api/v1/soi/qradle/proof` |
| route | `qratum_fullstack_server.py` | `/api/v1/soi/aethernet/validators` |
| route | `qratum_fullstack_server.py` | `/api/v1/soi/aethernet/consensus` |
| route | `qratum_fullstack_server.py` | `/api/v1/soi/aethernet/zones` |
| route | `qratum_fullstack_server.py` | `/api/v1/soi/trajectory/health` |
| route | `qratum_fullstack_server.py` | `/api/v1/soi/metrics` |
| route | `quasic_viz/services/qubic_render/ws_server.py` | `/frames/latest` |
| route | `quasic_viz/services/qubic_render/ws_server.py` | `/frames/{timestamp}` |
| route | `quasim/api/server.py` | `/api/v1/dtwin/{twin_id}/simulate` |
| route | `services/qubic-render/api.py` | `/sequence` |
| route | `services/qubic-render/api.py` | `/status/{job_id}` |
| route | `services/qubic-render/server.py` | `/gpu-status` |
| route | `soi/telemetry/state-stream.py` | `/api/v1/soi/qradle/state` |
| route | `soi/telemetry/state-stream.py` | `/api/v1/soi/qradle/proof` |
| route | `soi/telemetry/state-stream.py` | `/api/v1/soi/aethernet/validators` |
| route | `soi/telemetry/state-stream.py` | `/api/v1/soi/aethernet/consensus` |
| route | `soi/telemetry/state-stream.py` | `/api/v1/soi/trajectory/health` |
| route | `soi/telemetry/state-stream.py` | `/api/v1/soi/zones/stats` |
| route | `soi/telemetry/state-stream.py` | `/api/v1/soi/verticals` |
