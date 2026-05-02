FROM nvidia/cuda:12.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy service code
COPY . .

# Copy qubic-viz
COPY ../../qubic-viz /app/qubic-viz

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')"

# Run server
CMD ["python3", "-m", "uvicorn", "server:create_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]
