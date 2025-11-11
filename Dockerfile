# Use Python 3.11 slim image for minimal footprint
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-demo.txt .
RUN pip install --no-cache-dir -r requirements-demo.txt

# Copy demo script and profiles
COPY quasim_spacex_demo.py .
COPY configs/meco_profiles/ ./configs/meco_profiles/

# Default command runs Falcon 9 demo
CMD ["python", "quasim_spacex_demo.py", "--profile", "configs/meco_profiles/spacex_f9_stage1.json"]
