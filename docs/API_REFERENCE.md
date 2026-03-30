# QRATUM Platform API Reference

## Overview

The QRATUM Platform provides RESTful APIs for interacting with:

- **QRADLE**: Deterministic execution engine
- **QRATUM Platform**: 14 vertical AI modules
- **QRATUM-ASI**: Superintelligence orchestration layer (simulation mode)

All APIs return JSON and use standard HTTP status codes.

## Base URLs

- Development: `http://localhost:8000`
- Staging: `https://staging.qratum.io`
- Production: `https://api.qratum.io`

## Authentication

Currently authentication is handled via API keys (to be implemented).

```
Authorization: Bearer <your-api-key>
```

---

## QRADLE API

### Create and Execute Contract

Creates an immutable contract and executes it with full audit trail.

```http
POST /api/v1/qradle/contract
```

**Request Body:**

```json
{
  "operation": "add",
  "inputs": {
    "a": 5,
    "b": 3
  },
  "user_id": "user_123"
}
```

**Response:**

```json
{
  "success": true,
  "contract_id": "c7f3a8b2-1234-5678-90ab-cdef12345678",
  "status": "completed",
  "outputs": {
    "result": 8
  },
  "execution_time": 0.002
}
```

### Create Checkpoint

Creates a rollback checkpoint of current system state.

```http
POST /api/v1/system/checkpoint
```

**Request Body:**

```json
{
  "description": "Before major update"
}
```

**Response:**

```json
{
  "success": true,
  "checkpoint_id": "ckpt_abc123",
  "timestamp": 1703260800.0,
  "merkle_proof": "a3f5e2..."
}
```

### Get System Proof

Returns cryptographic proof of current system state.

```http
GET /api/v1/system/proof
```

**Response:**

```json
{
  "merkle_root": "7b3f2a...",
  "chain_length": 1247,
  "integrity_verified": true,
  "timestamp": 1703260800.0
}
```

### Get Audit Trail

Retrieves audit trail for all or specific contract.

```http
GET /api/v1/audit/trail?contract_id=<optional-contract-id>
```

**Response:**

```json
{
  "success": true,
  "events": [
    {
      "event_type": "contract_created",
      "data": {
        "contract_id": "c7f3a8b2...",
        "operation": "add",
        "user_id": "user_123",
        "timestamp": 1703260795.0
      },
      "previous_hash": "5a2c1f...",
      "hash": "9d4e3b..."
    }
  ],
  "count": 42
}
```

---

## QRATUM Platform API

### Execute Vertical Operation

Executes an operation on a specific vertical module.

```http
POST /api/v1/platform/execute
```

**Request Body:**

```json
{
  "vertical": "juris",
  "operation": "legal_reasoning",
  "parameters": {
    "facts": "Party A breached contract with Party B",
    "area_of_law": "contract"
  },
  "user_id": "user_123"
}
```

**Response:**

```json
{
  "success": true,
  "contract_id": "p8e2d4c1...",
  "result": {
    "issue": "Whether Party A breached contract",
    "rule": "Contract law principles apply",
    "application": "Based on facts...",
    "conclusion": "Party A likely breached contract"
  }
}
```

### List Available Verticals

Returns all available vertical modules.

```http
GET /api/v1/verticals
```

**Response:**

```json
{
  "success": true,
  "verticals": [
    {
      "name": "juris",
      "status": "active",
      "capabilities": "Legal AI & Compliance"
    },
    {
      "name": "vitra",
      "status": "active",
      "capabilities": "Bioinformatics & Drug Discovery"
    }
  ],
  "count": 14
}
```

---

## Vertical Modules

### JURIS - Legal AI

**Operations:**

- `legal_reasoning`: IRAC legal analysis
- `contract_analysis`: Contract review and risk identification
- `compliance_check`: Regulatory compliance checking
- `litigation_prediction`: Outcome prediction

**Example:**

```json
{
  "vertical": "juris",
  "operation": "compliance_check",
  "parameters": {
    "regulation": "GDPR",
    "activity": "data_processing"
  }
}
```

### VITRA - Bioinformatics

**Operations:**

- `sequence_analysis`: DNA/RNA/Protein analysis
- `drug_screening`: Drug candidate evaluation
- `molecular_dynamics`: Molecular simulation
- `adme_prediction`: Pharmacokinetics modeling

**Example:**

```json
{
  "vertical": "vitra",
  "operation": "sequence_analysis",
  "parameters": {
    "sequence": "ATGCGATCG...",
    "type": "DNA"
  }
}
```

### ECORA - Climate & Energy

**Operations:**

- `climate_projection`: Climate modeling
- `carbon_footprint`: Carbon emission analysis
- `energy_optimization`: Grid optimization
- `renewable_assessment`: Site evaluation

**Example:**

```json
{
  "vertical": "ecora",
  "operation": "carbon_footprint",
  "parameters": {
    "activity": "manufacturing",
    "energy_kwh": 10000
  }
}
```

### CAPRA - Financial Risk

**Operations:**

- `option_pricing`: Black-Scholes pricing
- `risk_assessment`: VaR/CVaR calculation
- `portfolio_optimization`: Portfolio optimization
- `credit_risk`: Credit analysis

**Example:**

```json
{
  "vertical": "capra",
  "operation": "option_pricing",
  "parameters": {
    "S": 100,
    "K": 100,
    "T": 1.0,
    "r": 0.05,
    "sigma": 0.2
  }
}
```

### SENTRA - Aerospace & Defense

**Operations:**

- `ballistic_trajectory`: Trajectory simulation
- `orbit_propagation`: Orbital mechanics
- `rcs_analysis`: Radar cross-section
- `mission_planning`: Mission optimization

### NEURA - Neuroscience & BCI

**Operations:**

- `eeg_analysis`: EEG signal processing
- `snn_simulation`: Spiking neural networks
- `connectivity_mapping`: Brain connectivity
- `bci_decoding`: BCI signal decoding

### FLUXA - Supply Chain

**Operations:**

- `route_optimization`: Vehicle routing
- `demand_forecasting`: Demand prediction
- `inventory_optimization`: Inventory management
- `resilience_analysis`: Network analysis

### SPECTRA - Spectrum Management

**Operations:**

- `spectrum_analysis`: RF spectrum analysis
- `interference_detection`: Interference detection
- `frequency_allocation`: Frequency optimization

### AEGIS - Cybersecurity

**Operations:**

- `vulnerability_scan`: Security scanning
- `threat_detection`: Threat analysis
- `compliance_check`: Security compliance

### LOGOS - Education & Training

**Operations:**

- `learning_path`: Personalized learning
- `knowledge_assessment`: Knowledge testing
- `skill_gap_analysis`: Skill analysis

### SYNTHOS - Materials Science

**Operations:**

- `predict_properties`: Material properties
- `crystal_structure`: Structure analysis
- `composite_design`: Composite design

### TERAGON - Geospatial Intelligence

**Operations:**

- `route_optimization`: Route planning
- `terrain_analysis`: Terrain analysis
- `spatial_pattern`: Pattern detection

### HELIX - Genomic Medicine

**Operations:**

- `variant_analysis`: Genetic variant analysis
- `risk_prediction`: Disease risk prediction
- `pharmacogenomics`: Drug response analysis

### NEXUS - Cross-Domain Intelligence

**Operations:**

- `multi_domain_synthesis`: Cross-domain synthesis
- `cross_domain_inference`: Domain bridging
- `emergent_pattern`: Pattern detection

---

## QRATUM-ASI API

### Get ASI Status

Returns current status of ASI orchestrator.

```http
GET /api/v1/asi/status
```

**Response:**

```json
{
  "mode": "simulation",
  "crsi_enabled": false,
  "safety_level": "routine",
  "components": {
    "q_reality": "active",
    "q_mind": "active",
    "q_evolve": "disabled",
    "q_will": "disabled",
    "q_forge": "active"
  }
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "success": false,
  "error": "Error message describing what went wrong",
  "code": "ERROR_CODE"
}
```

**Common Error Codes:**

- `400`: Bad Request - Invalid parameters
- `401`: Unauthorized - Invalid API key
- `403`: Forbidden - Insufficient permissions
- `404`: Not Found - Resource not found
- `429`: Too Many Requests - Rate limit exceeded
- `500`: Internal Server Error - Server error
- `503`: Service Unavailable - Service temporarily down

---

## Rate Limits

- Free tier: 100 requests/hour
- Standard tier: 1000 requests/hour
- Enterprise tier: Unlimited

---

## SDKs

### Python

```python
from qratum_sdk import QRATUMClient

client = QRATUMClient(api_key="your-api-key")

# Execute QRADLE contract
result = client.qradle.execute_contract(
    operation="add",
    inputs={"a": 5, "b": 3}
)

# Execute vertical operation
result = client.platform.execute(
    vertical="juris",
    operation="legal_reasoning",
    parameters={
        "facts": "Party A breached contract",
        "area_of_law": "contract"
    }
)
```

### TypeScript

```typescript
import { QRATUMClient } from '@qratum/sdk';

const client = new QRATUMClient({ apiKey: 'your-api-key' });

// Execute contract
const result = await client.qradle.executeContract({
  operation: 'add',
  inputs: { a: 5, b: 3 }
});
```

---

## Support

- Documentation: <https://docs.qratum.io>
- API Status: <https://status.qratum.io>
- Support: <support@qratum.io>
- GitHub: <https://github.com/robertringler/QRATUM>
