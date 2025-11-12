"""
QuASIM×QuNimbus Phase VII: Quantum-Economic Activation Layer

Full live Quantum-Economic Network (QEN) activation with:
- Quantum Market Protocol (QMP) with live liquidity partners
- Dynamic Φ-Valuation Engine (η_ent → real-time price metrics)
- Decentralized Verification Ledger (DVL) for Φ_QEVF + compliance attestations
- Quantum Market Daemon (QMD) and Φ-tokenization framework
- 6-region orchestration mesh (Americas, EU, MENA, APAC, Polar, Orbit)
- Extended continuous-compliance (ISO 27001, ITAR, GDPR)
"""

from quasim.qunimbus.phaseVII.dvl_ledger import DVLLedger
from quasim.qunimbus.phaseVII.qmp_activation import QMPActivation
from quasim.qunimbus.phaseVII.trust_kernel import TrustKernel
from quasim.qunimbus.phaseVII.valuation_engine import ValuationEngine

__version__ = "1.0.0-phaseVII-activation"

__all__ = [
    "QMPActivation",
    "ValuationEngine",
    "DVLLedger",
    "TrustKernel",
]
