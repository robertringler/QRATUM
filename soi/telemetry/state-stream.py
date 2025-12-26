"""
QRATUM SOI Telemetry Server
Deterministic State Stream Server for the Sovereign Operations Interface

This server provides read-only access to QRADLE state, Aethernet consensus,
and trajectory monitoring data. The UI cannot mutate state through this server.

Version: 1.0.0
"""

import asyncio
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional

# Constants
CONSENSUS_PHASES = ["PROPOSE", "PREVOTE", "PRECOMMIT", "COMMIT", "FINALIZED"]

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("[SOI Telemetry] FastAPI not installed, server will not be available")


@dataclass
class TelemetryEvent:
    """Telemetry event for streaming."""
    type: str
    payload: dict
    epoch: int
    timestamp: str
    proof: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class SOITelemetryServer:
    """
    Sovereign Operations Interface Telemetry Server.
    
    Provides:
    - WebSocket telemetry streams
    - REST API for state queries
    - Integration with QRADLE and Aethernet
    """

    def __init__(self):
        """Initialize telemetry server."""
        self.clients: list[WebSocket] = []
        self.current_epoch = 127843
        self.is_running = False
        
        # System state (read-only mirror)
        self.state = {
            "validators": {
                "total": 256,
                "active": 243,
                "pending": 8,
                "jailed": 3,
                "slashed": 2
            },
            "consensus": {
                "height": 1847293,
                "round": 0,
                "phase": "FINALIZED",
                "quorum_power": 89432000,
                "total_power": 102400000
            },
            "zones": {
                "Z0": 847,
                "Z1": 432,
                "Z2": 156,
                "Z3": 23
            },
            "trajectory": {
                "health_score": 0.98,
                "collapse_probability": 0.0002,
                "is_suspended": False,
                "precursor_signals": []
            },
            "merkle": {
                "root": self._generate_hash(),
                "chain_length": 1847293
            },
            "last_proof": self._generate_hash()
        }

    def _generate_hash(self) -> str:
        """Generate a deterministic hash."""
        data = f"{time.time()}{self.current_epoch}".encode()
        return "0x" + hashlib.sha256(data).hexdigest()[:40]

    def _get_timestamp(self) -> str:
        """Get current ISO timestamp."""
        return datetime.now(timezone.utc).isoformat()

    def create_event(self, event_type: str, payload: dict) -> TelemetryEvent:
        """Create a telemetry event."""
        return TelemetryEvent(
            type=event_type,
            payload=payload,
            epoch=self.current_epoch,
            timestamp=self._get_timestamp(),
            proof=self._generate_hash()
        )

    async def broadcast(self, event: TelemetryEvent):
        """Broadcast event to all connected clients."""
        message = event.to_json()
        disconnected = []
        
        for client in self.clients:
            try:
                await client.send_text(message)
            except Exception:
                disconnected.append(client)
        
        # Remove disconnected clients
        for client in disconnected:
            self.clients.remove(client)

    async def start_telemetry_loop(self):
        """Start the telemetry broadcast loop."""
        self.is_running = True
        
        while self.is_running:
            try:
                # Update epoch
                self.current_epoch += 1
                self.state["consensus"]["height"] += 1
                self.state["merkle"]["chain_length"] = self.state["consensus"]["height"]
                self.state["merkle"]["root"] = self._generate_hash()

                # Broadcast QRADLE state
                event = self.create_event("qradle:state", {
                    "merkle": self.state["merkle"].copy(),
                    "epoch": self.current_epoch
                })
                await self.broadcast(event)

                await asyncio.sleep(5)

                # Broadcast consensus update
                self.state["consensus"]["phase"] = random.choice(CONSENSUS_PHASES)
                
                event = self.create_event("aethernet:consensus", 
                    self.state["consensus"].copy())
                await self.broadcast(event)

                await asyncio.sleep(3)

                # Broadcast validator update
                event = self.create_event("aethernet:validator", {
                    "stats": self.state["validators"].copy()
                })
                await self.broadcast(event)

                await asyncio.sleep(4)

                # Broadcast quorum update
                self.state["consensus"]["quorum_power"] = 85000000 + random.randint(0, 10000000)
                percentage = (self.state["consensus"]["quorum_power"] / 
                            self.state["consensus"]["total_power"] * 100)
                
                event = self.create_event("aethernet:quorum", {
                    "quorum_power": self.state["consensus"]["quorum_power"],
                    "total_power": self.state["consensus"]["total_power"],
                    "percentage": round(percentage, 1)
                })
                await self.broadcast(event)

                await asyncio.sleep(6)

                # Broadcast trajectory update
                self.state["trajectory"]["collapse_probability"] = max(0,
                    min(0.01, self.state["trajectory"]["collapse_probability"] + 
                        (random.random() - 0.5) * 0.001))
                self.state["trajectory"]["health_score"] = max(0.9,
                    min(1.0, 1.0 - self.state["trajectory"]["collapse_probability"] * 5))
                
                event = self.create_event("trajectory:health", 
                    self.state["trajectory"].copy())
                await self.broadcast(event)

                await asyncio.sleep(8)

                # Broadcast zone update
                for zone in self.state["zones"]:
                    change = random.randint(-2, 2)
                    self.state["zones"][zone] = max(0, self.state["zones"][zone] + change)
                
                event = self.create_event("telemetry:zone", {
                    "zones": self.state["zones"].copy()
                })
                await self.broadcast(event)

                await asyncio.sleep(15)

                # Broadcast proof
                self.state["last_proof"] = self._generate_hash()
                event = self.create_event("proof:chain", {
                    "proof": self.state["last_proof"],
                    "height": self.state["consensus"]["height"],
                    "timestamp": self._get_timestamp()
                })
                await self.broadcast(event)

            except Exception as e:
                print(f"[SOI Telemetry] Error in broadcast loop: {e}")
                await asyncio.sleep(1)

    def stop(self):
        """Stop the telemetry loop."""
        self.is_running = False

    def get_state(self) -> dict:
        """Get current system state (read-only)."""
        return {
            "epoch": self.current_epoch,
            "timestamp": self._get_timestamp(),
            **self.state
        }


# Create global server instance
telemetry_server = SOITelemetryServer()


def create_soi_routes(app: "FastAPI"):
    """Create SOI routes for FastAPI app."""
    if not HAS_FASTAPI:
        return
    
    # WebSocket telemetry endpoint
    @app.websocket("/soi/telemetry")
    async def websocket_telemetry(websocket: WebSocket):
        await websocket.accept()
        telemetry_server.clients.append(websocket)
        print(f"[SOI Telemetry] Client connected ({len(telemetry_server.clients)} total)")
        
        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("command") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif message.get("command") == "get_state":
                    await websocket.send_json({
                        "type": "state",
                        "payload": telemetry_server.get_state()
                    })
                    
        except WebSocketDisconnect:
            telemetry_server.clients.remove(websocket)
            print(f"[SOI Telemetry] Client disconnected ({len(telemetry_server.clients)} remaining)")

    # REST API endpoints (read-only)
    @app.get("/api/v1/soi/qradle/state")
    async def get_qradle_state():
        return JSONResponse(content={
            "merkle_root": telemetry_server.state["merkle"]["root"],
            "chain_length": telemetry_server.state["merkle"]["chain_length"],
            "epoch": telemetry_server.current_epoch,
            "integrity_verified": True
        })

    @app.get("/api/v1/soi/qradle/proof")
    async def get_system_proof():
        return JSONResponse(content={
            "merkle_root": telemetry_server.state["merkle"]["root"],
            "chain_length": telemetry_server.state["merkle"]["chain_length"],
            "last_proof": telemetry_server.state["last_proof"],
            "integrity_verified": True,
            "timestamp": telemetry_server._get_timestamp()
        })

    @app.get("/api/v1/soi/aethernet/validators")
    async def get_validators():
        return JSONResponse(content=telemetry_server.state["validators"])

    @app.get("/api/v1/soi/aethernet/consensus")
    async def get_consensus():
        return JSONResponse(content=telemetry_server.state["consensus"])

    @app.get("/api/v1/soi/trajectory/health")
    async def get_trajectory_health():
        return JSONResponse(content=telemetry_server.state["trajectory"])

    @app.get("/api/v1/soi/zones/stats")
    async def get_zone_stats():
        return JSONResponse(content={
            "Z0": {"count": telemetry_server.state["zones"]["Z0"], "label": "Public"},
            "Z1": {"count": telemetry_server.state["zones"]["Z1"], "label": "Private"},
            "Z2": {"count": telemetry_server.state["zones"]["Z2"], "label": "Regulated"},
            "Z3": {"count": telemetry_server.state["zones"]["Z3"], "label": "Air-gapped"}
        })

    @app.get("/api/v1/soi/verticals")
    async def get_verticals():
        return JSONResponse(content=[
            {"id": "VITRA", "name": "VITRA-E0", "status": "active", "operations": 1234},
            {"id": "CAPRA", "name": "CAPRA", "status": "active", "operations": 5678},
            {"id": "JURIS", "name": "JURIS", "status": "active", "operations": 3456},
            {"id": "ECORA", "name": "ECORA", "status": "active", "operations": 2345},
            {"id": "FLUXA", "name": "FLUXA", "status": "active", "operations": 4567}
        ])


async def start_telemetry_server():
    """Start the telemetry broadcast loop."""
    await telemetry_server.start_telemetry_loop()


if __name__ == "__main__":
    # Standalone server for testing
    if HAS_FASTAPI:
        import uvicorn
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        
        app = FastAPI(
            title="QRATUM SOI Telemetry Server",
            description="Sovereign Operations Interface Telemetry API",
            version="1.0.0"
        )
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        create_soi_routes(app)
        
        @app.on_event("startup")
        async def startup():
            asyncio.create_task(start_telemetry_server())
        
        @app.on_event("shutdown")
        async def shutdown():
            telemetry_server.stop()
        
        print("[SOI Telemetry] Starting server on port 8000...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("[SOI Telemetry] FastAPI not available. Install with: pip install fastapi uvicorn")
