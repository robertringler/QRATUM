# Network Topology

## Quantum-Aware Network Architecture

### Network Segmentation

1. **Quantum Plane:** QKD-secured, deterministic routing
2. **Control Plane:** Zero-trust, CAC/PIV authentication
3. **Bulk Data Plane:** High-throughput, QUIC over SRv6

### Protocols

- **QKD:** Quantum Key Distribution for unconditional security
- **QUIC:** Low-latency reliable transport
- **SRv6:** Segment Routing for quantum-aware paths
- **mTLS:** Mutual TLS with PIV cards

### Topology

- **Core:** 400G spine switches (CLOS fabric)
- **Quantum:** Dedicated photonic mesh (512:1 fanout)
- **Storage:** NVMe-oF over 200G RDMA

### Security

- **Zero Trust:** All communications authenticated and encrypted
- **OPA Gatekeeper:** Policy enforcement at network layer
- **Fortinet:** Perimeter security and DPI
- **CAC/PIV:** Hardware-based authentication

### Bandwidth

- **Inter-Rack:** 1.6 Tbps aggregate
- **Intra-Rack:** 5.12 Tbps via NVLink C2C
- **External:** 800G uplinks to data center fabric
