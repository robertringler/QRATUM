#!/usr/bin/env python3
"""
QRATUM Platform API Server

FastAPI-based REST API for QRATUM platform.
Provides endpoints for all 14 verticals and cross-domain synthesis.
"""

import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, '/app')

from qratum.platform.api import (
    QRATUMAPIService,
    APIRequest,
    APIResponse,
    SynthesisRequest,
    SynthesisResponse
)

# Initialize API service
api_service = QRATUMAPIService()

# Create FastAPI app
app = FastAPI(
    title="QRATUM Platform API",
    description="Sovereign AI Infrastructure with 14 Vertical Modules",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Pydantic models for API
class VerticalTaskRequest(BaseModel):
    vertical: str
    task: str
    parameters: Dict[str, Any]
    requester_id: str = "api_client"
    safety_level: str = "ROUTINE"
    authorized: bool = False

class MultiVerticalSynthesisRequest(BaseModel):
    query: str
    verticals: List[str]
    parameters: Optional[Dict[str, Any]] = None
    strategy: str = "deductive"
    requester_id: str = "api_client"

# Routes

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "QRATUM Platform API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "verticals": "/api/v1/verticals",
            "execute": "/api/v1/vertical/execute",
            "synthesis": "/api/v1/synthesis/execute"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    status = api_service.get_health_status()
    return JSONResponse(content=status)

@app.get("/api/v1/verticals")
async def list_verticals():
    """List all available vertical modules."""
    verticals = api_service.list_verticals()
    return {"verticals": verticals, "count": len(verticals)}

@app.get("/api/v1/vertical/{vertical_id}")
async def get_vertical_info(vertical_id: str):
    """Get information about a specific vertical."""
    info = api_service.get_vertical_info(vertical_id.upper())
    if not info:
        raise HTTPException(status_code=404, detail=f"Vertical '{vertical_id}' not found")
    return {"vertical_id": vertical_id.upper(), **info}

@app.post("/api/v1/vertical/execute")
async def execute_vertical_task(request: VerticalTaskRequest):
    """Execute a task on a specific vertical module."""
    api_request = APIRequest(
        vertical=request.vertical.upper(),
        task=request.task,
        parameters=request.parameters,
        requester_id=request.requester_id,
        safety_level=request.safety_level,
        authorized=request.authorized
    )
    
    response = api_service.execute_vertical_task(api_request)
    
    if not response.success:
        raise HTTPException(status_code=400, detail=response.error)
    
    return {
        "success": response.success,
        "data": response.data,
        "execution_time": response.execution_time,
        "output_hash": response.output_hash,
        "checkpoint_id": response.checkpoint_id,
        "safety_disclaimer": response.safety_disclaimer
    }

@app.post("/api/v1/synthesis/execute")
async def execute_synthesis(request: MultiVerticalSynthesisRequest):
    """Execute cross-domain synthesis across multiple verticals."""
    synthesis_request = SynthesisRequest(
        query=request.query,
        verticals=[v.upper() for v in request.verticals],
        parameters=request.parameters,
        strategy=request.strategy,
        requester_id=request.requester_id
    )
    
    response = api_service.execute_synthesis(synthesis_request)
    
    return {
        "success": response.success,
        "chain_id": response.chain_id,
        "query": response.query,
        "verticals_used": response.verticals_used,
        "final_conclusion": response.final_conclusion,
        "confidence": response.confidence,
        "provenance_hash": response.provenance_hash,
        "reasoning_nodes_count": response.reasoning_nodes_count,
        "execution_time": response.execution_time
    }

@app.get("/api/v1/synthesis/verify/{chain_id}")
async def verify_reasoning_chain(chain_id: str):
    """Verify integrity of a reasoning chain."""
    result = api_service.verify_reasoning_chain(chain_id)
    return result

@app.get("/api/v1/stats")
async def get_statistics():
    """Get system statistics."""
    stats = api_service.get_statistics()
    return stats

# Run server
if __name__ == "__main__":
    port = int(os.getenv("API_PORT", "8000"))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
