FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# Install system dependencies (build essentials and libs for lxml/ssl)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        make \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
        libffi-dev \
        libssl-dev \
    ca-certificates \
    curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Also bake a copy of the EPL Football-Data cache outside /app/data so it isn't
# shadowed by the persistent volume in production (Render mounts /app/data).
RUN mkdir -p /app/baked \
    && cp data/football_data_epl_2025_2026.json /app/baked/football_data_epl_2025_2026.json || true

# Expose port (informational)
EXPOSE 8000

# Run the application; honor Render's PORT env var
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]