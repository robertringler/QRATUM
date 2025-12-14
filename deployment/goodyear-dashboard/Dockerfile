# Goodyear Quantum Tire Visualization Suite
# Base: NVIDIA CUDA 12.2 with Ubuntu 22.04
# QuASIM Integration with Claude 4.5 Opus for Real-Time Dashboards

FROM nvidia/cuda:12.2.0-base-ubuntu22.04

LABEL maintainer="QuASIM Team <quasim@example.com>"
LABEL description="Goodyear Quantum Tire Visualization Dashboard with QuASIM Integration"
LABEL version="1.0.0"

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=all
ENV DASH_DEBUG=false
ENV DASH_HOST=0.0.0.0
ENV DASH_PORT=8050

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    curl \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libice6 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Create application directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY launch_dashboard.py /app/launch_dashboard.py
COPY assets/ /app/assets/

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/cache

# Set permissions
RUN chmod +x /app/launch_dashboard.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8050/health || exit 1

# Expose dashboard port
EXPOSE 8050

# Run as non-root user for security
RUN groupadd -r dashuser && useradd -r -g dashuser dashuser
RUN chown -R dashuser:dashuser /app
USER dashuser

# Entrypoint
ENTRYPOINT ["python", "/app/launch_dashboard.py"]
