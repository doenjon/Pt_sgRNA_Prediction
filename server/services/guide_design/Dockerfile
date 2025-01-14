FROM python:3.7-slim

WORKDIR /app

# Install system dependencies and clean up in the same layer
RUN apt-get update && apt-get install -y \
    build-essential \
    bwa \
    gcc \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create and set permissions for temp directories
RUN mkdir -p /var/tmp && chmod 777 /var/tmp

# Copy requirements first for better caching
COPY server/services/guide_design/requirements.txt .

# Install packages with specific versions for Python 3.7
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Create package directories
COPY pt_guide_design /app/packages/pt_guide_design/
COPY sgrna_scorer /app/packages/sgrna_scorer/

# Set PYTHONPATH environment variable
ENV PYTHONPATH=/app/packages:$PYTHONPATH

# Install the packages in development mode
RUN cd /app/packages/sgrna_scorer && pip install -e . && \
    cd /app/packages/pt_guide_design && pip install -e .

# Copy the service code
COPY server/services/guide_design/service.py .

# Add these lines to make CRISPOR script executable
RUN chmod +x /app/packages/pt_guide_design/crisporWebsite/crispor.py

CMD ["python", "service.py"]