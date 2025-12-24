# QRADLE Foundation - Production Docker Image
# Deterministic execution engine with cryptographic guarantees
#
# Version: 1.0.0
# Compliance: DO-178C, CMMC Level 3, ISO 27001

FROM python:3.11-slim AS base

# Security hardening
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r qradle && useradd -r -g qradle qradle

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy QRADLE source
COPY qradle/ ./qradle/
COPY setup.py ./

# Install QRADLE
RUN pip install --no-cache-dir -e .

# Create data directories
RUN mkdir -p /app/data /app/checkpoints /app/logs \
    && chown -R qradle:qradle /app

# Switch to non-root user
USER qradle

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import qradle; print('OK')"

# Default command
CMD ["python", "-c", "from qradle import DeterministicEngine; engine = DeterministicEngine(); print('QRADLE Engine Ready'); import time; time.sleep(infinity)"]

# Metadata
LABEL maintainer="QRATUM Team"
LABEL version="1.0.0"
LABEL description="QRADLE - Deterministic Execution Engine"
LABEL compliance="DO-178C,CMMC-L3,ISO27001"
