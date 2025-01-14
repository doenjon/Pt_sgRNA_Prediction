FROM python:3.7-slim

WORKDIR /app

# Add build arg for cache busting when needed
ARG GIT_COMMIT=latest
ARG BUILDKIT_INLINE_CACHE=1

# Copy requirements first and install dependencies
COPY server/services/guide_design/requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    bwa \
    gcc \
    python3-dev \
    git \
    bedtools \
    samtools \
    tabix \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create and set permissions for temp directories
RUN mkdir -p /var/tmp && chmod 777 /var/tmp

# Copy setup files first to cache dependency installation
COPY pt_guide_design/setup.py /app/packages/pt_guide_design/
COPY sgrna_scorer/setup.py /app/packages/sgrna_scorer/

# Install packages in development mode
RUN cd /app/packages/sgrna_scorer && pip install -e . && \
    cd /app/packages/pt_guide_design && pip install -e .

# Copy source code
# Use .dockerignore to exclude unnecessary files
COPY pt_guide_design /app/packages/pt_guide_design/
COPY sgrna_scorer /app/packages/sgrna_scorer/

ENV PYTHONPATH=/app/packages:$PYTHONPATH

# Verify CRISPOR files
RUN ls -la /app/packages/pt_guide_design/crisporWebsite && \
    chmod +x /app/packages/pt_guide_design/crisporWebsite/crispor.py

COPY server/services/guide_design/service.py .

CMD ["python", "service.py"]