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

# Bake league fixture caches into /app/baked (outside /app/data which Render mounts)
RUN mkdir -p /app/baked
# Use COPY so files are embedded even if cp would be skipped
COPY data/football_data_epl_2025_2026.json /app/baked/
COPY data/football_data_PL_2025_2026.json /app/baked/
COPY data/football_data_BL1_2025_2026.json /app/baked/
COPY data/football_data_FL1_2025_2026.json /app/baked/
COPY data/football_data_SA_2025_2026.json /app/baked/
COPY data/football_data_PD_2025_2026.json /app/baked/

# Expose port (informational)
EXPOSE 8000

# Run the application; honor Render's PORT env var
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]