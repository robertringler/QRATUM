# Data Flow

Understanding how data flows through the QRATUM platform.

## Request Processing

### VQE Computation Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant A as API
    participant V as VQE Module
    participant Q as Quantum Backend
    participant O as Classical Optimizer
    
    C->>A: POST /vqe {molecule, config}
    A->>V: compute_energy(params)
    
    loop Until Converged
        V->>Q: Execute Circuit
        Q->>V: Measurement Results
        V->>O: Energy Value
        O->>V: New Parameters
    end
    
    V->>A: VQEResult
    A->>C: {energy, iterations, converged}
```

### QAOA Optimization Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant A as API
    participant Q as QAOA Module
    participant B as Backend
    participant O as Optimizer
    
    C->>A: POST /qaoa {graph, config}
    A->>Q: solve_maxcut(edges)
    
    Q->>Q: Build Problem Hamiltonian
    Q->>Q: Initialize Parameters
    
    loop Optimization Loop
        Q->>B: Execute QAOA Circuit
        B->>Q: Bitstring Distribution
        Q->>Q: Compute Expectation
        Q->>O: Energy Value
        O->>Q: Updated (gamma, beta)
    end
    
    Q->>Q: Sample Best Solution
    Q->>A: QAOAResult
    A->>C: {partition, cut_value, ratio}
```

## Data Transformations

### Molecule to Quantum Circuit

```
H₂ Molecule
    │
    ▼
┌─────────────────────┐
│ Molecular Geometry  │  bond_length = 0.735 Å
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Hamiltonian (PySCF) │  H = Σ h_pq a†_p a_q + ...
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Jordan-Wigner Map   │  Fermions → Qubits
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Qubit Hamiltonian   │  H = Σ c_i P_i (Paulis)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Parameterized Ansatz│  |ψ(θ)⟩ = U(θ)|0⟩
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Quantum Circuit     │  Executable circuit
└─────────────────────┘
```

### Graph to QAOA Circuit

```
Graph G = (V, E)
    │
    ▼
┌─────────────────────┐
│ Problem Hamiltonian │  C = Σ (1-Z_iZ_j)/2
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Mixing Hamiltonian  │  B = Σ X_i
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ QAOA Circuit        │  U_B U_C ... U_B U_C |+⟩^n
│ (p layers)          │
└─────────────────────┘
```

## Execution Backends

### Backend Selection Flow

```mermaid
graph TD
    A[QuantumConfig] --> B{backend_type?}
    B -->|simulator| C[Qiskit Aer]
    B -->|ibmq| D[IBM Quantum]
    B -->|cuquantum| E[NVIDIA cuQuantum]
    
    C --> F[Local Execution]
    D --> G[Cloud Queue]
    E --> H[GPU Execution]
    
    F --> I[Results]
    G --> I
    H --> I
```

### Simulator Pipeline

```
Parameters θ
    │
    ▼
┌─────────────────────┐
│ Build Circuit       │  Qiskit QuantumCircuit
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Compile to Backend  │  Transpile for Aer
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Execute (N shots)   │  Sample measurements
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Aggregate Results   │  Compute expectation
└─────────────────────┘
```

## Distributed Processing

### QuNimbus Data Flow

```mermaid
graph TB
    subgraph "Region A"
        C1[Client] --> API1[API Gateway]
        API1 --> W1[Worker Pods]
    end
    
    subgraph "Region B"
        C2[Client] --> API2[API Gateway]
        API2 --> W2[Worker Pods]
    end
    
    subgraph "Control Plane"
        QN[QuNimbus Controller]
        DB[(State Store)]
        Q[(Job Queue)]
    end
    
    API1 --> Q
    API2 --> Q
    W1 --> QN
    W2 --> QN
    QN --> DB
```

### Job Processing Flow

```
Job Submission
    │
    ▼
┌─────────────────────┐
│ API Gateway         │  Authenticate, validate
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Job Queue (Redis)   │  Priority queue
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Scheduler           │  Resource allocation
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌───────┐   ┌───────┐
│Worker │   │Worker │    Execute in parallel
│  1    │   │  2    │
└───┬───┘   └───┬───┘
    │           │
    └─────┬─────┘
          │
          ▼
┌─────────────────────┐
│ Result Aggregation  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Client Response     │
└─────────────────────┘
```

## Observability Data Flow

### Metrics Pipeline

```
Application
    │
    │ Prometheus client
    ▼
┌─────────────────────┐
│ /metrics endpoint   │  Expose metrics
└─────────┬───────────┘
          │
          │ HTTP scrape (30s)
          ▼
┌─────────────────────┐
│ Prometheus Server   │  Store time series
└─────────┬───────────┘
          │
          │ PromQL queries
          ▼
┌─────────────────────┐
│ Grafana Dashboard   │  Visualize
└─────────────────────┘
```

### Logging Pipeline

```
Application Log
    │
    │ stdout/stderr
    ▼
┌─────────────────────┐
│ Container Runtime   │  Capture logs
└─────────┬───────────┘
          │
          │ Fluentd/Promtail
          ▼
┌─────────────────────┐
│ Loki                │  Log aggregation
└─────────┬───────────┘
          │
          │ LogQL queries
          ▼
┌─────────────────────┐
│ Grafana             │  Search & visualize
└─────────────────────┘
```

## Data Persistence

### State Management

```mermaid
graph LR
    A[Application] --> B{Data Type}
    B -->|Ephemeral| C[Redis Cache]
    B -->|Persistent| D[PostgreSQL]
    B -->|Files| E[S3/GCS]
    B -->|Metrics| F[Prometheus TSDB]
    B -->|Logs| G[Loki]
```

### Data Retention

| Data Type | Storage | Retention |
|-----------|---------|-----------|
| Job results | PostgreSQL | 90 days |
| Session cache | Redis | 24 hours |
| Metrics | Prometheus | 15 days |
| Logs | Loki | 30 days |
| Audit logs | S3 | 7 years |

## Security Data Flow

### Authentication

```mermaid
sequenceDiagram
    participant U as User
    participant G as Gateway
    participant A as Auth Service
    participant V as Vault
    
    U->>G: Request + API Key
    G->>A: Validate Key
    A->>V: Get Signing Key
    V->>A: Key Material
    A->>A: Verify JWT
    A->>G: Claims
    G->>G: Check RBAC
```

### Encryption

```
Data in Transit           Data at Rest
────────────────          ─────────────
                          
Client ──TLS 1.3──▶ API   PostgreSQL (AES-256)
API ──mTLS──▶ Service     Redis (Encrypted)
Service ──TLS──▶ DB       S3 (SSE-KMS)
```
