# Build stage
FROM python:3.9-slim AS builder

WORKDIR /app

# Set environment variables for pip
ENV TMPDIR=/var/tmp \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create and set permissions for temp directories
RUN mkdir -p /var/tmp && \
    chmod 777 /var/tmp

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    zlib1g-dev \
    bwa \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY server/services/guide_design/requirements.txt .
RUN --mount=type=tmpfs,target=/var/tmp \
    pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.9-slim

WORKDIR /app

# Set environment variables
ENV TMPDIR=/var/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create and set permissions for temp directories
RUN mkdir -p /var/tmp && \
    chmod 777 /var/tmp

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    bwa \
    && rm -rf /var/lib/apt/lists/*

# Copy the packages
COPY sgrna_scorer /app/packages/sgrna_scorer
COPY pt_guide_design /app/packages/pt_guide_design

# Install packages in development mode with temp directory handling
RUN --mount=type=tmpfs,target=/var/tmp \
    pip install --no-cache-dir -e /app/packages/sgrna_scorer && \
    pip install --no-cache-dir -e /app/packages/pt_guide_design

# Copy the service code
COPY server/services/guide_design/service.py .

CMD ["python", "service.py"]