# Architecture Overview

```mermaid
graph TD
    CPU[72-core Grace CPU Complex]
    GPU[Blackwell Tensor GPU Cluster]
    HBM[LPDDR5x Unified Memory]
    NVLink[NVLink-C2C Fabric]
    NIC[ConnectX-7 SmartNIC]
    NVMe[NVMe Gen5 Controller]
    CPU -->|Coherent AXI/CHI| NVLink
    GPU -->|Coherent Fabric| NVLink
    NVLink --> HBM
    NVLink --> NIC
    NVLink --> NVMe
```

The GB10 superchip couples a high-core-count Grace-inspired CPU complex with a Blackwell-style GPU cluster through a coherent NVLink-C2C fabric. The CPU delivers Arm v9 SVE2 compute with advanced power-state management, while the GPU exposes SIMT scheduling, 5th generation tensor cores, and ray-tracing acceleration.

The unified fabric allows every agent to operate in a shared virtual address space, enabling low-latency data exchange between AI and quantum simulation workloads.
