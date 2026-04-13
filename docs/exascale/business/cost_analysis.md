# QRATUM ExaScale Cost Analysis

## Executive Summary

**Total Project Cost:** $4.52 Billion  
**Capital Expenditure (CapEx):** $4.16 Billion  
**Operational Expenditure (OpEx, 5-year):** $360 Million  
**Cost per ExaFLOPS:** $2.13 Billion (vs. Frontier: $1.8B)

The QRATUM ExaScale Initiative represents a $4.5B investment in next-generation supercomputing infrastructure, targeting TOP500 #1 position by 2028.

## Capital Expenditure (CapEx) Breakdown

### 1. Compute Hardware ($2.28B)

| Component | Quantity | Unit Cost | Total Cost |
|-----------|----------|-----------|------------|
| **NVIDIA H100 GPUs** | 50,000 | $30,000 | **$1,500,000,000** |
| AMD EPYC 9654 CPUs | 12,500 | $12,000 | $150,000,000 |
| DDR5 Memory (2TB/node) | 12,500 | $8,000 | $100,000,000 |
| NVMe Gen5 (245 TB/node) | 49.2 PB | $200k/PB | $9,840,000 |
| CN-QES Chassis | 6,250 | $15,000 | $93,750,000 |
| NVLink Switches | 25,000 | $2,000 | $50,000,000 |
| **Subtotal** | | | **$1,903,590,000** |

**Note:** GPU pricing assumes DOE/DOD priority allocation discount (30% off list price of $43k).

### 2. Networking ($384M)

| Component | Quantity | Unit Cost | Total Cost |
|-----------|----------|-----------|------------|
| **AetherFabric-X NICs** | 25,000 | $5,000 | **$125,000,000** |
| **AetherFabric-X Switches** | 1,037 | $250,000 | **$259,250,000** |
| - Leaf Switches (64-port) | 781 | $200,000 | $156,200,000 |
| - Spine Switches (128-port) | 256 | $400,000 | $102,400,000 |
| Optical Transceivers (800G) | 50,000 | $800 | $40,000,000 |
| Fiber Cables (OM5) | 200 km | $100/m | $20,000,000 |
| **Subtotal** | | | **$384,250,000** |

### 3. Cooling System ($350M)

| Component | Quantity | Unit Cost | Total Cost |
|-----------|----------|-----------|------------|
| Dilution Refrigerators | 5 | $20,000,000 | $100,000,000 |
| Cooling Distribution Units | 8 | $5,000,000 | $40,000,000 |
| Pumps (40 L/min, redundant) | 12,500 | $5,000 | $62,500,000 |
| Dielectric Coolant (Novec 7100) | 2,000 m³ | $25,000/m³ | $50,000,000 |
| Cold Plates (GPU/CPU) | 100,000 | $500 | $50,000,000 |
| Piping & Manifolds | - | - | $47,500,000 |
| **Subtotal** | | | **$350,000,000** |

### 4. Power Distribution ($280M)

| Component | Quantity | Unit Cost | Total Cost |
|-----------|----------|-----------|------------|
| Utility Transformers (15 MVA) | 3 | $5,000,000 | $15,000,000 |
| Medium Voltage Switchgear | 8 | $8,000,000 | $64,000,000 |
| Busway Distribution (5000A) | 80 | $500,000 | $40,000,000 |
| Rack PDUs (48V conversion) | 6,250 | $12,000 | $75,000,000 |
| UPS System (50 MVA, Li-ion) | 1 | $50,000,000 | $50,000,000 |
| Diesel Generators (15 MW) | 4 | $8,000,000 | $32,000,000 |
| Electrical Installation | - | - | $4,000,000 |
| **Subtotal** | | | **$280,000,000** |

### 5. Facility Construction ($800M)

| Component | Cost |
|-----------|------|
| Site Preparation | $50,000,000 |
| Building Construction (20,000 m²) | $300,000,000 |
| Raised Floor (seismic-rated) | $40,000,000 |
| HVAC (non-compute areas) | $60,000,000 |
| Fire Suppression (Novec 1230) | $30,000,000 |
| Security Systems | $20,000,000 |
| Monitoring & BMS | $15,000,000 |
| Commissioning & Testing | $35,000,000 |
| Contingency (15%) | $120,000,000 |
| **Subtotal** | **$800,000,000** |

### 6. Storage Infrastructure ($450M)

| Component | Capacity | Unit Cost | Total Cost |
|-----------|----------|-----------|------------|
| Lustre Parallel FS | 50 PB | $5,000/TB | $250,000,000 |
| - Object Storage Servers | 100 | $500,000 | $50,000,000 |
| - Metadata Servers | 20 | $300,000 | $6,000,000 |
| - InfiniBand Network | - | - | $44,000,000 |
| Tape Archive (LTO-9) | 500 PB | $50/TB | $25,000,000 |
| - Tape Libraries | 10 | $5,000,000 | $50,000,000 |
| Burst Buffer (NVMe-oF) | 10 PB | $10,000/TB | $100,000,000 |
| **Subtotal** | | | **$450,000,000** |

### 7. Quantum Integration ($100M)

| Component | Quantity | Unit Cost | Total Cost |
|-----------|----------|-----------|------------|
| PhotonQ PTAQ-1000 QPUs | 16 | $5,000,000 | $80,000,000 |
| Quantum Control Modules | 16 | $500,000 | $8,000,000 |
| NVLink-Q Bridges | 16 | $250,000 | $4,000,000 |
| Cryogenic Infrastructure | - | - | $8,000,000 |
| **Subtotal** | | | **$100,000,000** |

### Total CapEx Summary

| Category | Cost | % of Total |
|----------|------|------------|
| Compute Hardware | $1,903M | 45.7% |
| Networking | $384M | 9.2% |
| Cooling | $350M | 8.4% |
| Power | $280M | 6.7% |
| Facility | $800M | 19.2% |
| Storage | $450M | 10.8% |
| Quantum | $100M | 2.4% |
| **Total CapEx** | **$4,157M** | **100%** |

## Operational Expenditure (OpEx, Annual)

### 1. Electricity ($42M/year)

**Power Consumption:**

- Compute: 35 MW (average, 80% utilization)
- Cooling: 6.8 MW
- Overhead: 3.1 MW
- **Total:** 44.9 MW average

**Annual Energy:**

- 44.9 MW × 8,760 hours = 393,324 MWh/year

**Cost:**

- Rate: $0.08/kWh (negotiated industrial rate)
- Annual cost: 393,324 × $80 = **$31,465,920**
- Demand charges: $10M/year (peak capacity)
- **Total:** $42M/year

### 2. Personnel ($12M/year)

| Role | Headcount | Avg Salary | Total Cost |
|------|-----------|------------|------------|
| System Administrators | 20 | $150,000 | $3,000,000 |
| Network Engineers | 10 | $160,000 | $1,600,000 |
| HPC Scientists | 15 | $180,000 | $2,700,000 |
| Facility Operators | 25 | $80,000 | $2,000,000 |
| Security/Support | 15 | $100,000 | $1,500,000 |
| Management | 5 | $250,000 | $1,250,000 |
| **Total (with benefits)** | **90** | | **$12,050,000** |

### 3. Maintenance & Spare Parts ($8M/year)

| Component | Annual Cost | Notes |
|-----------|-------------|-------|
| GPU Replacements | $4,500,000 | 1.5% failure rate |
| NIC/Switch Repairs | $1,500,000 | 2% failure rate |
| Cooling Maintenance | $800,000 | Pumps, chillers |
| Facility Maintenance | $500,000 | HVAC, fire systems |
| Storage Media | $700,000 | Tape refresh |
| **Total** | **$8,000,000** | |

### 4. Software Licenses ($3M/year)

| Software | Annual Cost |
|----------|-------------|
| NVIDIA CUDA Toolkit (enterprise) | $500,000 |
| Lustre Professional Support | $1,000,000 |
| Monitoring (Nagios, Grafana, etc.) | $300,000 |
| Development Tools (compilers, debuggers) | $500,000 |
| Operating Systems (RHEL, Ubuntu Pro) | $700,000 |
| **Total** | **$3,000,000** |

### 5. Network & Cloud Services ($2M/year)

| Service | Annual Cost |
|---------|-------------|
| Internet Connectivity (100 Gbps) | $1,200,000 |
| Cloud Backup (for critical data) | $500,000 |
| VPN & Security Services | $300,000 |
| **Total** | **$2,000,000** |

### 6. Insurance & Compliance ($5M/year)

| Item | Annual Cost |
|------|-------------|
| Property Insurance | $3,000,000 |
| Liability Insurance | $1,000,000 |
| ITAR Compliance Audits | $500,000 |
| Environmental Compliance | $500,000 |
| **Total** | **$5,000,000** |

### Total OpEx Summary (Annual)

| Category | Annual Cost |
|----------|-------------|
| Electricity | $42,000,000 |
| Personnel | $12,000,000 |
| Maintenance | $8,000,000 |
| Software | $3,000,000 |
| Network | $2,000,000 |
| Insurance | $5,000,000 |
| **Total OpEx** | **$72,000,000/year** |

**5-Year OpEx:** $72M × 5 = **$360M**

## Total Cost of Ownership (TCO)

**5-Year TCO:**

- CapEx: $4,157M
- OpEx (5 years): $360M
- **Total:** $4,517M

**10-Year TCO:**

- CapEx: $4,157M
- OpEx (10 years): $720M
- Hardware Refresh (Year 5): $1,000M (partial upgrade)
- **Total:** $5,877M

## Financing Strategy

### Funding Sources

| Source | Amount | Terms |
|--------|--------|-------|
| **DOE ASCR Program** | $1,500M | Grant (no repayment) |
| **DOD HPCMP** | $1,000M | Grant (no repayment) |
| **NSF Major Research Instrumentation** | $500M | Grant (no repayment) |
| **Industry Partnerships** | $500M | Cost-sharing (NVIDIA, AMD, Intel) |
| **University Contribution** | $157M | Institutional funds |
| **Debt Financing (bonds)** | $500M | 3.5% interest, 20-year term |
| **Total** | **$4,157M** | |

### Return on Investment (ROI)

**Revenue Streams:**

1. **Academic Research:** $10M/year (grant overhead recovery)
2. **Industry Partnerships:** $30M/year (compute time sales)
3. **Cloud Services:** $20M/year (QRATUM-as-a-Service)
4. **IP Licensing:** $5M/year (patents, software)
5. **Total Revenue:** $65M/year

**ROI Calculation:**

- Annual revenue: $65M
- Annual OpEx: $72M
- **Net:** -$7M/year (subsidized by grants)

**Payback Period:** N/A (non-profit research infrastructure)

**Societal ROI:** Estimated $50B+ in scientific/economic impact over 10 years (drug discovery, climate modeling, materials science).

## Cost Comparison (Per ExaFLOPS)

| System | Cost | ExaFLOPS | $/ExaFLOPS |
|--------|------|----------|------------|
| Frontier (ORNL) | $600M | 1.194 | $502M |
| Aurora (ANL) | $500M | 1.012 | $494M |
| **QRATUM** | **$4,157M** | **2.125** | **$1,957M** |

**Note:** QRATUM's higher cost reflects:

1. Custom networking (AetherFabric-X): +$150M
2. Quantum integration: +$100M
3. Advanced cooling: +$100M
4. Post-quantum security: +$50M

**Adjusted (excluding R&D):** $3,657M / 2.125 = **$1,721M/ExaFLOPS**

## Conclusion

The QRATUM ExaScale Initiative requires a $4.5B total investment, with $4.16B in capital expenditure and $72M annual operating costs. While the per-ExaFLOPS cost is higher than existing systems, QRATUM delivers:

1. **Performance Leadership:** TOP500 #1 by 2028
2. **Technological Innovation:** First exascale quantum-classical hybrid
3. **Long-Term Value:** 10-year lifespan with upgrade path to 10 ExaFLOPS
4. **Scientific Impact:** Enable breakthroughs in climate, health, and national security

---
**Document Version:** 1.0  
**Last Updated:** 2025-01-26  
**Classification:** Business Sensitive
