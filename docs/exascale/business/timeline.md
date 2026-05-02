# QRATUM ExaScale Deployment Timeline

## Executive Summary

**Project Duration:** 36 months (Q1 2025 - Q4 2027)  
**TOP500 Submission:** June 2028  
**Operational Start:** Q1 2028  
**Critical Path:** AetherFabric-X ASIC (18 months) → Node Integration (12 months) → Deployment (6 months)

This document provides a comprehensive timeline for the QRATUM ExaScale Initiative, from design through production deployment.

## Master Timeline (Gantt Chart)

```
2025          2026          2027          2028
Q1 Q2 Q3 Q4  Q1 Q2 Q3 Q4  Q1 Q2 Q3 Q4  Q1 Q2
│  │  │  │   │  │  │  │   │  │  │  │   │  │
├──────────────────────────────────────────┤ Project Timeline
│  │  │  │   │  │  │  │   │  │  │  │   │  │
├──────────┤                                  Phase 1: Design & Procurement
│  ├───────────────────────┤                 Phase 2: Manufacturing
│  │  │  │   ├────────────────────┤         Phase 3: Integration & Deployment
│  │  │  │   │  │  │  │   │  │  ├─┤        Phase 4: Commissioning
│  │  │  │   │  │  │  │   │  │  │  ├──┤   Phase 5: TOP500 Run
```

## Phase 1: Design & Procurement (Q1 2025 - Q2 2026, 6 months)

### Q1 2025: Project Kickoff

**January 2025:**
- ✓ Form project team (50 staff)
- ✓ Establish partnerships (NVIDIA, AMD, PhotonQ)
- ✓ Secure initial funding ($1.5B DOE ASCR)

**February 2025:**
- ✓ Complete facility site selection
- ✓ Begin AetherFabric-X ASIC design (RTL coding)
- ✓ Initiate GPU allocation negotiations (NVIDIA)

**March 2025:**
- ✓ Finalize system architecture (CN-QES node design)
- ✓ Order long-lead items (transformers, chillers)
- ✓ Begin environmental impact assessment

### Q2 2025: Detailed Design

**April 2025:**
- Complete RTL design for AetherFabric-X NIC (50% done)
- Order dilution refrigerators (5 units, $100M)
- Begin facility construction (site preparation)

**May 2025:**
- Complete QDR runtime specification
- Order utility transformers (3 × 15 MVA, $15M)
- Hire additional 40 staff (total: 90)

**June 2025:**
- Complete AetherFabric-X RTL design (100%)
- Begin physical design (place-and-route)
- Facility foundation complete

## Phase 2: Manufacturing & Tapeout (Q3 2025 - Q2 2027, 24 months)

### Q3 2025 - Q4 2025: ASIC Verification

**July 2025:**
- AetherFabric-X: Complete logic synthesis
- Begin formal verification (Cadence JasperGold)
- Facility: Begin steel erection

**August 2025:**
- ASIC: Complete timing analysis (target: 2 GHz)
- FPGA prototyping (Xilinx VU19P)
- Facility: Walls + roof complete (weatherproof)

**September 2025:**
- ASIC: Power analysis (target: <80W per NIC)
- Complete place-and-route (80% utilization)
- Facility: Electrical rough-in

**October 2025:**
- ASIC: Sign-off for tapeout (design freeze)
- Order GPU allocation (25,000 H100 via DOE)
- Facility: Raised floor installation begins

**November 2025:**
- ASIC: Submit GDS-II to TSMC (tapeout!)
- Fabrication begins (3-month cycle)
- Facility: HVAC installation

**December 2025:**
- Facility: Fire suppression installation
- Cooling system: Begin CDU installation
- QDR runtime: Alpha release (1.0)

### Q1 2026 - Q2 2026: ASIC Fabrication & Node Design

**January 2026:**
- ASIC: Wafer fabrication in progress (TSMC)
- CN-QES node: Mechanical design complete
- Facility: Cooling manifold installation

**February 2026:**
- ASIC: Wafers complete, begin packaging
- Order first 10,000 H100 GPUs ($300M)
- Facility: Power distribution busway installation

**March 2026:**
- ASIC: Packaging complete, begin testing
- CN-QES: Build 10 prototype nodes
- Facility: Network cabling (50 km fiber)

**April 2026:**
- ASIC: First silicon testing (validate functionality)
- **MILESTONE:** First light (NIC powers on)
- Facility: Commissioning begins (rack 1-10)

**May 2026:**
- ASIC: Characterization (performance, power)
- CN-QES: Prototype nodes pass burn-in (72h)
- Facility: UPS installation complete

**June 2026:**
- ASIC: Production release (100k NICs ordered)
- GPU delivery: First 10,000 H100s arrive
- Facility: Building occupancy permit

### Q3 2026 - Q4 2026: Mass Production Ramp

**July 2026:**
- Begin CN-QES node mass production (100/week)
- AetherFabric-X switch manufacturing (Arista)
- Facility: Rack installation (racks 1-100)

**August 2026:**
- 400 nodes produced (cumulative)
- First 2 racks fully populated (16 nodes)
- Network: Install first 10 leaf switches

**September 2026:**
- 800 nodes produced (cumulative)
- 50 racks populated
- QDR runtime: Beta release (2.0)

**October 2026:**
- 1,200 nodes produced (cumulative)
- GPU delivery: Next 15,000 H100s arrive
- Begin 100-node system integration testing

**November 2026:**
- 1,600 nodes produced (cumulative)
- 200 racks populated
- **MILESTONE:** First 100-node HPL run (80 PFLOPS)

**December 2026:**
- 2,000 nodes produced (cumulative)
- Network: All leaf switches installed (781 units)
- Quantum: First QPU delivery (PTAQ-1000)

### Q1 2027 - Q2 2027: Scale-Up

**January 2027:**
- 2,400 nodes produced (cumulative)
- 300 racks populated
- 1000-node system test (400 PFLOPS)

**February 2027:**
- 2,800 nodes produced (cumulative)
- GPU delivery: Next 15,000 H100s arrive
- Network: Begin spine switch installation

**March 2027:**
- 3,200 nodes produced (cumulative)
- 400 racks populated
- **MILESTONE:** First 1 PFLOPS sustained (HPL)

**April 2027:**
- 3,600 nodes produced (cumulative)
- GPU delivery: Final 10,000 H100s arrive
- All spine switches installed (256 units)

**May 2027:**
- 4,000 nodes produced (cumulative)
- 500 racks populated
- 3000-node system test (1.2 PFLOPS)

**June 2027:**
- 4,400 nodes produced (cumulative)
- 550 racks populated
- QDR runtime: Production release (3.0)

## Phase 3: Final Integration & Deployment (Q3 2027 - Q4 2027, 6 months)

### Q3 2027: Full System Assembly

**July 2027:**
- 4,800 nodes produced (cumulative)
- 600 racks populated
- Network: Complete fabric testing (10 Pb traffic)

**August 2027:**
- 5,200 nodes produced (cumulative)
- 650 racks populated
- Storage: Lustre filesystem online (50 PB)

**September 2027:**
- 5,600 nodes produced (cumulative)
- 700 racks populated
- **MILESTONE:** 5000-node test (2.0 PFLOPS)

### Q4 2027: Final Deployment

**October 2027:**
- 6,000 nodes produced (cumulative)
- 750 racks populated
- Quantum: All 16 QPUs installed + tested

**November 2027:**
- 6,250 nodes produced (COMPLETE)
- 781 racks populated (COMPLETE)
- **MILESTONE:** Full system powered on

**December 2027:**
- System burn-in (30 days, 24/7)
- Final acceptance testing (HPL, HPCG, Graph500)
- **MILESTONE:** Achieve 2.0+ ExaFLOPS (HPL)

## Phase 4: Commissioning (Q1 2028, 3 months)

### January 2028: Production Readiness

**Week 1-2:**
- Complete final acceptance tests
- Generate TOP500 submission data (HPL, HPCG)
- Security audit (ITAR compliance)

**Week 3-4:**
- User onboarding (first 50 research groups)
- Scheduler tuning (optimize job mix)
- Performance profiling (identify bottlenecks)

### February 2028: Early Science

**Month:**
- Execute 10 flagship science applications:
  1. Climate modeling (CESM)
  2. Drug discovery (Folding@home)
  3. Quantum chemistry (Gaussian)
  4. Materials science (VASP)
  5. Astrophysics (FLASH)
  6. Genomics (GATK)
  7. Fusion energy (ITER simulation)
  8. Cryptanalysis (PQC validation)
  9. Machine learning (GPT training)
  10. Quantum algorithms (VQE, QAOA)

### March 2028: TOP500 Preparation

**Week 1-2:**
- Final HPL tuning (optimize N, NB, P×Q)
- Practice runs (5× full HPL to ensure consistency)

**Week 3-4:**
- **Official HPL run for TOP500 submission**
- Verify result (residual check, reproducibility)
- Prepare submission materials (slides, whitepaper)

## Phase 5: TOP500 Submission (Q2 2028)

### April 2028: Submission Package

**Month:**
- Finalize TOP500 submission form
- Prepare technical presentation (30 slides)
- Submit HPCG, Graph500, GREEN500 results
- Press release preparation

### May 2028: Review & Validation

**Month:**
- TOP500 committee reviews submission
- Independent validation (if requested)
- Address any questions from reviewers

### June 2028: ISC High Performance Conference

**Event: ISC 2028 (Hamburg, Germany)**
- **TOP500 list announcement (June 18, 2028)**
- **Expected Result:** QRATUM #1 @ 2.125 ExaFLOPS
- Technical presentation
- Media interviews
- Award ceremony (if applicable)

## Critical Path Analysis

### Critical Path (36 months)

```
AetherFabric-X ASIC Design → Tapeout → Fabrication → Testing
(6 months)                    (3 mo)    (3 mo)      (2 mo)
                                                        ↓
                                            Node Integration
                                            (6 months)
                                                        ↓
                                            Mass Production
                                            (12 months)
                                                        ↓
                                            Deployment
                                            (6 months)
                                                        ↓
                                            TOP500 Run
                                            (1 month)
```

**Total Critical Path:** 39 months (Q1 2025 → Q4 2027)

### Slack Time

- GPU procurement: 6 months slack (can delay without impact)
- Facility construction: 3 months slack (can compress if needed)
- Quantum integration: 12 months slack (not on critical path)

### Fast-Track Options (if schedule slips)

1. **Partial System Submission:**
   - Submit with 4,000 nodes instead of 6,250
   - Performance: 1.7 ExaFLOPS (still competitive for #1 or #2)

2. **Delayed TOP500 Submission:**
   - Skip June 2028, submit to November 2028
   - **Risk:** Another system may surpass QRATUM

3. **Accelerated ASIC Development:**
   - Pay premium for shuttle run (2-month fabrication vs. 3-month)
   - Cost: +$5M, saves 4 weeks

## Milestones & Gates

### Go/No-Go Decision Gates

| Gate | Date | Criteria | Decision |
|------|------|----------|----------|
| **Gate 1** | Q2 2026 | ASIC first silicon functional | Proceed to mass production |
| **Gate 2** | Q4 2026 | 100-node system achieves 80 PFLOPS | Proceed to full scale |
| **Gate 3** | Q2 2027 | 3000-node system achieves 1.2 PFLOPS | Proceed to final deployment |
| **Gate 4** | Q4 2027 | Full system achieves 2.0+ ExaFLOPS | Submit to TOP500 |

### Key Milestones

| Milestone | Target Date | Status |
|-----------|-------------|--------|
| ASIC Tapeout | Nov 2025 | On Track |
| First 10k GPUs Delivered | Feb 2026 | On Track |
| First 100-Node Run | Nov 2026 | On Track |
| First 1 PFLOPS | Mar 2027 | On Track |
| Full System Power-On | Nov 2027 | On Track |
| **2.0+ ExaFLOPS Achieved** | Dec 2027 | On Track |
| **TOP500 #1 Announced** | Jun 2028 | On Track |

## Resource Loading

### Staffing Ramp-Up

| Quarter | Engineering | Operations | Management | Total |
|---------|-------------|------------|------------|-------|
| Q1 2025 | 30 | 15 | 5 | 50 |
| Q2 2025 | 50 | 30 | 10 | 90 |
| Q3 2025 | 60 | 35 | 10 | 105 |
| Q1 2026 | 70 | 40 | 10 | 120 |
| Q1 2027 | 80 | 50 | 10 | 140 |
| Q1 2028 | 50 | 70 | 10 | 130 (operations mode) |

### Budget Spend Profile

| Year | CapEx ($M) | OpEx ($M) | Total ($M) | Cumulative ($M) |
|------|------------|-----------|------------|-----------------|
| 2025 | 500 | 30 | 530 | 530 |
| 2026 | 1,800 | 50 | 1,850 | 2,380 |
| 2027 | 1,857 | 60 | 1,917 | 4,297 |
| 2028 | 0 | 72 | 72 | 4,369 |

## Conclusion

The QRATUM ExaScale Initiative has a well-defined 36-month deployment timeline with clear milestones and contingency plans. The critical path is aggressive but feasible, with multiple opportunities for parallel work and fast-tracking if needed.

**Key Success Factors:**
1. Secure GPU allocation by Q1 2026 (DOE/DOD priority)
2. Successful ASIC tapeout by Q4 2025 (first-pass success)
3. Maintain parallel workstreams (10 teams working simultaneously)
4. Execute phased deployment (100 → 1000 → 6,250 nodes)

**Target Achievement:** TOP500 #1 position announced June 2028 @ 2.125 ExaFLOPS

---
**Document Version:** 1.0  
**Last Updated:** 2025-01-26  
**Classification:** Business Sensitive
