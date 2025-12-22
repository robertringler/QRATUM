#!/usr/bin/env python3
"""
QRATUM Full Stack Server

Integrates QRADLE, QRATUM Platform (14 verticals), and QRATUM-ASI layers
into a unified production-ready deployment.
"""

import os
import sys
import time
import logging
from typing import Dict, Any, Optional
from flask import Flask, jsonify, request
from flask_cors import CORS

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import QRADLE
from qradle import QRADLEEngine

# Import QRATUM Platform
from qratum_platform.core import QRATUMPlatform, PlatformIntent, VerticalModule

# Import all verticals
from verticals import (
    JURISModule, VITRAModule, ECORAModule, CAPRAModule,
    SENTRAModule, NEURAModule, FLUXAModule, SPECTRAModule,
    AEGISModule, LOGOSModule, SYNTHOSModule, TERAGONModule,
    HELIXModule, NEXUSModule
)

# Import QRATUM-ASI
from qratum_asi.orchestrator import ASIOrchestrator

# Setup logging
logging.basicConfig(
    level=os.getenv("QRATUM_LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global state
qradle_engine: Optional[QRADLEEngine] = None
qratum_platform: Optional[QRATUMPlatform] = None
asi_orchestrator: Optional[ASIOrchestrator] = None

COMPONENT = os.getenv("QRATUM_COMPONENT", "full-stack")


def initialize_qradle() -> QRADLEEngine:
    """Initialize QRADLE engine."""
    logger.info("Initializing QRADLE engine...")
    engine = QRADLEEngine()
    
    # Register basic operations
    engine.register_operation("echo", lambda inputs: {"output": inputs.get("input")})
    engine.register_operation("add", lambda inputs: {"result": inputs.get("a", 0) + inputs.get("b", 0)})
    
    logger.info("QRADLE engine initialized successfully")
    return engine


def initialize_platform() -> QRATUMPlatform:
    """Initialize QRATUM platform with all 14 verticals."""
    logger.info("Initializing QRATUM platform...")
    platform = QRATUMPlatform()
    
    # Register all 14 verticals
    verticals = {
        VerticalModule.JURIS: JURISModule(),
        VerticalModule.VITRA: VITRAModule(),
        VerticalModule.ECORA: ECORAModule(),
        VerticalModule.CAPRA: CAPRAModule(),
        VerticalModule.SENTRA: SENTRAModule(),
        VerticalModule.NEURA: NEURAModule(),
        VerticalModule.FLUXA: FLUXAModule(),
        VerticalModule.SPECTRA: SPECTRAModule(),
        VerticalModule.AEGIS: AEGISModule(),
        VerticalModule.LOGOS: LOGOSModule(),
        VerticalModule.SYNTHOS: SYNTHOSModule(),
        VerticalModule.TERAGON: TERAGONModule(),
        VerticalModule.HELIX: HELIXModule(),
        VerticalModule.NEXUS: NEXUSModule(),
    }
    
    for vertical_type, module in verticals.items():
        platform.register_module(vertical_type, module)
        logger.info(f"Registered vertical: {vertical_type.value}")
    
    logger.info("QRATUM platform initialized with 14 verticals")
    return platform


def initialize_asi() -> ASIOrchestrator:
    """Initialize QRATUM-ASI orchestrator."""
    logger.info("Initializing QRATUM-ASI orchestrator...")
    mode = os.getenv("ASI_MODE", "simulation")
    orchestrator = ASIOrchestrator(mode=mode)
    logger.info(f"QRATUM-ASI initialized in {mode} mode")
    return orchestrator


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "component": COMPONENT,
        "timestamp": time.time(),
        "qradle": qradle_engine is not None,
        "platform": qratum_platform is not None,
        "asi": asi_orchestrator is not None
    })


@app.route('/api/v1/qradle/contract', methods=['POST'])
def create_contract():
    """Create and execute a QRADLE contract."""
    try:
        data = request.json
        operation = data.get("operation")
        inputs = data.get("inputs", {})
        user_id = data.get("user_id", "anonymous")
        
        # Create contract
        contract = qradle_engine.create_contract(
            operation=operation,
            inputs=inputs,
            user_id=user_id
        )
        
        # Execute contract
        execution = qradle_engine.execute_contract(contract)
        
        return jsonify({
            "success": True,
            "contract_id": contract.contract_id,
            "status": execution.status.value,
            "outputs": execution.outputs,
            "execution_time": execution.execution_time
        })
    except Exception as e:
        logger.error(f"Contract execution error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/platform/execute', methods=['POST'])
def execute_vertical():
    """Execute a vertical module operation."""
    try:
        data = request.json
        vertical = data.get("vertical")
        operation = data.get("operation")
        parameters = data.get("parameters", {})
        user_id = data.get("user_id", "anonymous")
        
        # Create intent
        intent = PlatformIntent(
            vertical=VerticalModule(vertical),
            operation=operation,
            parameters=parameters,
            user_id=user_id
        )
        
        # Create and execute contract
        contract = qratum_platform.create_contract(intent)
        result = qratum_platform.execute_contract(contract.contract_id)
        
        return jsonify({
            "success": True,
            "contract_id": contract.contract_id,
            "result": result
        })
    except Exception as e:
        logger.error(f"Vertical execution error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/asi/status', methods=['GET'])
def asi_status():
    """Get ASI orchestrator status."""
    try:
        status = asi_orchestrator.get_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"ASI status error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/system/proof', methods=['GET'])
def system_proof():
    """Get cryptographic proof of system state."""
    try:
        proof = qradle_engine.get_system_proof()
        return jsonify(proof)
    except Exception as e:
        logger.error(f"System proof error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/system/checkpoint', methods=['POST'])
def create_checkpoint():
    """Create a system checkpoint."""
    try:
        data = request.json or {}
        description = data.get("description", "Manual checkpoint")
        
        checkpoint = qradle_engine.create_checkpoint(description)
        
        return jsonify({
            "success": True,
            "checkpoint_id": checkpoint.checkpoint_id,
            "timestamp": checkpoint.timestamp,
            "merkle_proof": checkpoint.merkle_proof
        })
    except Exception as e:
        logger.error(f"Checkpoint creation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/audit/trail', methods=['GET'])
def audit_trail():
    """Get audit trail."""
    try:
        contract_id = request.args.get("contract_id")
        events = qradle_engine.get_audit_trail(contract_id)
        
        return jsonify({
            "success": True,
            "events": events,
            "count": len(events)
        })
    except Exception as e:
        logger.error(f"Audit trail error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/verticals', methods=['GET'])
def list_verticals():
    """List all available verticals."""
    verticals = [
        {
            "name": v.value,
            "status": "active",
            "capabilities": qratum_platform.modules.get(v, {})
        }
        for v in VerticalModule
    ]
    return jsonify({
        "success": True,
        "verticals": verticals,
        "count": len(verticals)
    })


def main():
    """Main entry point."""
    global qradle_engine, qratum_platform, asi_orchestrator
    
    logger.info(f"Starting QRATUM Full Stack Server (component: {COMPONENT})")
    
    # Initialize components based on configuration
    if COMPONENT in ("full-stack", "qradle"):
        qradle_engine = initialize_qradle()
    
    if COMPONENT in ("full-stack", "platform"):
        qratum_platform = initialize_platform()
    
    if COMPONENT in ("full-stack", "asi"):
        asi_orchestrator = initialize_asi()
    
    # Start server
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Server starting on {host}:{port}")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
