# QRATUM Container Topology

_Phase_: U1 Canonical Architecture  
_Generated_: 2026-04-30

## Container Files

| File | Purpose | USER | HEALTHCHECK |
|------|---------|------|-------------|
| `Dockerfile` | Main QRATUM image | ✅ 1001 (added GAP-CTR-NOROOT) | ✅ (added GAP-CTR-HEALTH) |
| `Dockerfile.production` | Production image | TBD | TBD |
| `Dockerfile.sandbox-platform` | Sandbox platform | ✅ 1001 (added) | ✅ (added) |
| `Dockerfile.sandbox-qradle` | Qradle sandbox | ✅ 1001 (added) | ✅ (added) |
| `docker-compose.yml` | Dev orchestration | inherited | inherited |
| `docker-compose.production.yml` | Prod orchestration | inherited | inherited |

## Service Graph

```
docker-compose.production.yml
├── qratum-api         (port 8080) ← qratum_fullstack_server.py
├── qratum-dashboard   (port 3000) ← dashboard/
├── quasim-engine      (port 8090) ← quasim_master_all.py
├── xenon-live         (port 8081) ← xenon_live_server.py
└── qunimbus           (port 8092) ← quasim/qunimbus/
```

## Security Hardening Applied

- All three sandbox Dockerfiles now run as non-root (USER 1001)
- HEALTHCHECK added to all three sandbox/main Dockerfiles
- Base image `curl` installed for healthcheck probe
- Container FS remains writable for mounted volumes (non-root UID 1001 owns `/app`)

## Remaining TODOs

- `Dockerfile.production` needs USER + HEALTHCHECK (deferred, lower risk than sandbox images)
- Resource limits (`--memory`, `--cpus`) should be set in `docker-compose.production.yml`
- Image signing + SBOM attestation: planned for CI/CD hardening phase
