"""Orchestrator service for routing requests to appropriate model servers."""

import logging
from typing import Dict, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="QRATUM Orchestrator", version="1.0.0")

# Routing configuration
ROUTING: Dict[str, str] = {
    "default": "http://model-server:8080/predict",
    "code": "http://model-server-code:8080/predict",
    "chat": "http://model-server-chat:8080/predict",
}


class RouteRequest(BaseModel):
    """Request payload for routing."""

    query: str
    intent: Optional[str] = "default"


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "orchestrator"}


@app.post("/route")
def route(payload: RouteRequest):
    """Route requests to appropriate model server based on intent.

    Args:
        payload: Route request with query and intent

    Returns:
        Response from the target model server

    Raises:
        HTTPException: If routing fails or target is unavailable
    """
    intent = payload.intent or "default"
    target = ROUTING.get(intent, ROUTING["default"])

    logger.info(f"Routing request with intent '{intent}' to {target}")

    try:
        # Forward request to target model server
        resp = requests.post(
            target,
            json={"query": payload.query, "intent": intent},
            headers={"Authorization": "Bearer SECRET_TOKEN"},
            timeout=30,
        )
        resp.raise_for_status()

        result = resp.json()
        logger.info(
            f"Successfully routed to {target}, confidence: {result.get('confidence', 'N/A')}"
        )
        return result

    except requests.exceptions.Timeout:
        logger.error(f"Timeout connecting to {target}")
        raise HTTPException(status_code=504, detail="gateway timeout")
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error to {target}")
        raise HTTPException(status_code=503, detail="service unavailable")
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error from {target}: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error routing to {target}: {e}")
        raise HTTPException(status_code=500, detail="internal server error")


@app.get("/routes")
def list_routes():
    """List available routing configurations."""
    return {"routes": ROUTING}


@app.get("/")
def root():
    """Root endpoint with service information."""
    return {
        "service": "QRATUM Orchestrator",
        "version": "1.0.0",
        "endpoints": ["/health", "/route", "/routes"],
    }


if __name__ == "__main__":
    logger.info("Starting QRATUM Orchestrator on port 8000")
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
