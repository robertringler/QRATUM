# QRATUM Full Stack Platform - API Reference

## Overview

QRATUM provides a comprehensive REST API for accessing all 14 vertical AI modules and cross-domain synthesis capabilities. The API is designed for sovereign, on-premises deployment with complete data control.

## Base URL

```
http://localhost:8000/api/v1
```

For production with SSL:
```
https://your-domain.com/api/v1
```

## Authentication

### API Key Authentication

Include API key in request headers:

```bash
curl -H "X-API-Key: your-api-key" \
     http://localhost:8000/api/v1/verticals
```

### mTLS (Mutual TLS)

For high-security environments, client certificates are required.

## Core Endpoints

### Health Check

**GET** `/health`

Check system health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-01T00:00:00Z",
  "verticals_available": 14,
  "qradle_executions": 1234,
  "reasoning_chains": 56,
  "merkle_chain_valid": true
}
```

### List Verticals

**GET** `/api/v1/verticals`

Get all available vertical modules.

**Response:**
```json
{
  "verticals": [
    {
      "vertical_id": "JURIS",
      "name": "Legal & Compliance",
      "tasks": ["analyze_contract", "legal_reasoning", ...],
      "safety_disclaimer": "⚖️ LEGAL DISCLAIMER: ..."
    },
    ...
  ],
  "count": 14
}
```

## Vertical Execution

### Execute Vertical Task

**POST** `/api/v1/vertical/execute`

Execute a task on a specific vertical module.

**Request Body:**
```json
{
  "vertical": "JURIS",
  "task": "analyze_contract",
  "parameters": {
    "contract_text": "This Agreement is made...",
    "analysis_depth": "comprehensive"
  },
  "requester_id": "user123",
  "safety_level": "ROUTINE",
  "authorized": true
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "vertical": "JURIS",
    "task": "analyze_contract",
    "result": "Contract analysis complete",
    "insights": [
      "Insight 1: Termination clause unclear",
      "Insight 2: Liability cap well-defined"
    ],
    "recommendations": [
      "Clarify termination conditions",
      "Review indemnification clause"
    ]
  },
  "execution_time": 0.523,
  "output_hash": "a1b2c3d4...",
  "checkpoint_id": "checkpoint_abc123",
  "safety_disclaimer": "⚖️ LEGAL DISCLAIMER: This analysis is for informational purposes only."
}
```

## All 14 Verticals

### 1. JURIS - Legal & Compliance

**Tasks:**
- `analyze_contract`: Analyze contract for risks and compliance
- `legal_reasoning`: Apply legal reasoning frameworks (IRAC/CRAC)
- `predict_litigation`: Predict litigation outcomes
- `check_compliance`: Check regulatory compliance

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/vertical/execute \
  -H "Content-Type: application/json" \
  -d '{
    "vertical": "JURIS",
    "task": "analyze_contract",
    "parameters": {
      "contract_text": "Contract text here...",
      "jurisdiction": "US-NY"
    }
  }'
```

### 2. VITRA - Healthcare & Life Sciences

**Tasks:**
- `diagnose_condition`: Medical diagnosis support
- `drug_interaction`: Check drug interactions
- `clinical_decision`: Clinical decision support
- `research_analysis`: Medical research analysis

### 3. ECORA - Climate & Environment

**Tasks:**
- `climate_model`: Climate impact modeling
- `sustainability_analysis`: Sustainability assessment
- `resource_optimization`: Resource optimization
- `impact_assessment`: Environmental impact analysis

### 4. CAPRA - Finance & Economics

**Tasks:**
- `risk_assessment`: Financial risk assessment
- `fraud_detection`: Fraud detection and analysis
- `market_analysis`: Market trend analysis
- `portfolio_optimization`: Portfolio optimization

### 5. SENTRA - Security & Defense

**Tasks:**
- `threat_detection`: Threat detection and analysis
- `vulnerability_analysis`: Security vulnerability analysis
- `strategic_planning`: Strategic security planning
- `risk_modeling`: Security risk modeling

### 6. NEURA - Cognitive Science

**Tasks:**
- `behavioral_analysis`: Behavioral pattern analysis
- `cognitive_modeling`: Cognitive process modeling
- `mental_health_support`: Mental health support (informational)
- `human_factors`: Human factors analysis

### 7. FLUXA - Supply Chain

**Tasks:**
- `optimize_route`: Route optimization
- `demand_forecast`: Demand forecasting
- `inventory_management`: Inventory optimization
- `supplier_analysis`: Supplier risk analysis

### 8. CHRONA - Temporal Reasoning

**Tasks:**
- `time_series_analysis`: Time series analysis
- `predictive_modeling`: Predictive modeling
- `scenario_planning`: Future scenario planning
- `trend_detection`: Trend detection and analysis

### 9. GEONA - Geospatial

**Tasks:**
- `spatial_analysis`: Spatial data analysis
- `route_optimization`: Geographic route optimization
- `terrain_modeling`: Terrain modeling
- `location_intelligence`: Location intelligence

### 10. FUSIA - Energy & Materials

**Tasks:**
- `grid_optimization`: Energy grid optimization
- `materials_discovery`: Materials science discovery
- `energy_modeling`: Energy system modeling
- `fusion_research`: Fusion research support

### 11. STRATA - Social Systems

**Tasks:**
- `policy_analysis`: Policy impact analysis
- `social_impact`: Social impact assessment
- `governance_modeling`: Governance system modeling
- `equity_assessment`: Equity and fairness assessment

### 12. VEXOR - Adversarial & Game Theory

**Tasks:**
- `strategic_game`: Strategic game analysis
- `adversarial_reasoning`: Adversarial reasoning
- `negotiation_analysis`: Negotiation strategy analysis
- `game_theory_modeling`: Game theory modeling

### 13. COHORA - Collaborative Intelligence

**Tasks:**
- `multi_agent_coordination`: Multi-agent coordination
- `collective_decision`: Collective decision making
- `swarm_optimization`: Swarm optimization
- `consensus_building`: Consensus building

### 14. ORBIA - Orbital & Space

**Tasks:**
- `orbital_mechanics`: Orbital mechanics calculation
- `satellite_ops`: Satellite operations planning
- `space_mission_planning`: Space mission planning
- `trajectory_optimization`: Trajectory optimization

## Cross-Domain Synthesis

### Execute Multi-Vertical Synthesis

**POST** `/api/v1/synthesis/execute`

Synthesize knowledge across multiple verticals.

**Request Body:**
```json
{
  "query": "How does climate change affect pharmaceutical supply chains?",
  "verticals": ["ECORA", "VITRA", "FLUXA"],
  "parameters": {
    "analysis_depth": "comprehensive",
    "time_horizon": "10_years"
  },
  "strategy": "causal",
  "requester_id": "analyst123"
}
```

**Response:**
```json
{
  "success": true,
  "chain_id": "reasoning_chain_123_456",
  "query": "How does climate change affect pharmaceutical supply chains?",
  "verticals_used": ["ECORA", "VITRA", "FLUXA"],
  "final_conclusion": {
    "synthesis_type": "multi_vertical_causal",
    "verticals_consulted": ["ECORA", "VITRA", "FLUXA"],
    "aggregated_insights": [
      "Insight 1 from ECORA",
      "Insight 2 from VITRA",
      "Insight 3 from FLUXA"
    ],
    "cross_domain_connections": [
      {
        "from_vertical": "ECORA",
        "to_vertical": "FLUXA",
        "connection_type": "causal_link",
        "strength": 0.85
      }
    ],
    "confidence_weighted_conclusion": "Multi-vertical analysis complete with 87.5% confidence"
  },
  "confidence": 0.875,
  "provenance_hash": "def456...",
  "reasoning_nodes_count": 3,
  "execution_time": 1.234
}
```

### Verify Reasoning Chain

**GET** `/api/v1/synthesis/verify/{chain_id}`

Verify cryptographic integrity of reasoning chain.

**Response:**
```json
{
  "chain_id": "reasoning_chain_123_456",
  "exists": true,
  "verified": true,
  "provenance_hash": "def456..."
}
```

## Safety Levels

All operations are classified by safety level:

| Level | Authorization | Use Cases |
|-------|---------------|-----------|
| `ROUTINE` | None required | Read operations, queries |
| `ELEVATED` | Logging + notification | Complex analysis |
| `SENSITIVE` | Single human approval | Configuration changes |
| `CRITICAL` | Multi-human approval | Self-improvement |
| `EXISTENTIAL` | Board + external | Architecture changes |

Set `safety_level` in your request:
```json
{
  "safety_level": "SENSITIVE",
  "authorized": true
}
```

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message",
  "status_code": 400,
  "error_type": "validation_error"
}
```

### Common HTTP Status Codes

- `200 OK`: Success
- `400 Bad Request`: Invalid parameters
- `401 Unauthorized`: Missing/invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

## Rate Limiting

Default limits:
- 1000 requests/minute per API key
- 10 synthesis requests/minute per API key

Headers included in response:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1609459200
```

## Statistics & Monitoring

### Get System Statistics

**GET** `/api/v1/stats`

**Response:**
```json
{
  "verticals": 14,
  "qradle": {
    "total_executions": 1234,
    "chain_length": 5678,
    "chain_valid": true
  },
  "reasoning_engine": {
    "total_chains": 56
  }
}
```

## SDK Examples

### Python

```python
import requests

# Execute vertical task
response = requests.post(
    "http://localhost:8000/api/v1/vertical/execute",
    json={
        "vertical": "JURIS",
        "task": "analyze_contract",
        "parameters": {"contract_text": "..."},
        "authorized": True
    }
)
result = response.json()
print(f"Analysis complete: {result['data']}")
```

### cURL

```bash
# List verticals
curl http://localhost:8000/api/v1/verticals

# Execute task
curl -X POST http://localhost:8000/api/v1/vertical/execute \
  -H "Content-Type: application/json" \
  -d '{"vertical":"CAPRA","task":"risk_assessment","parameters":{"asset":"AAPL"}}'

# Cross-domain synthesis
curl -X POST http://localhost:8000/api/v1/synthesis/execute \
  -H "Content-Type: application/json" \
  -d '{"query":"Optimize renewable energy grid","verticals":["ECORA","FUSIA","GEONA"]}'
```

## Support

- **API Issues**: https://github.com/robertringler/QRATUM/issues
- **Documentation**: https://docs.qratum.io
- **Email**: api-support@qratum.io
