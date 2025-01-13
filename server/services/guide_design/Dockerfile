FROM python:3.9-slim

WORKDIR /app

# Install system dependencies and clean up in the same layer
RUN apt-get update && apt-get install -y \
    build-essential \
    bwa \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create and set permissions for temp directories
RUN mkdir -p /var/tmp && chmod 777 /var/tmp

# Copy requirements first for better caching
COPY server/services/guide_design/requirements.txt .

# Install packages in smaller chunks with temporary directory configuration
ENV TMPDIR=/var/tmp
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DEFAULT_TIMEOUT=100

# Install base packages first
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir numpy pandas

# Install ML packages separately
RUN pip install --no-cache-dir scikit-learn==1.2.2
RUN pip install --no-cache-dir tensorflow==2.12.0

# Install remaining packages
RUN pip install --no-cache-dir biopython gffutils pybedtools tqdm pyfaidx

# Copy the packages with correct paths
COPY sgrna_scorer /app/packages/sgrna_scorer
COPY pt_guide_design /app/packages/pt_guide_design

# Install local packages
WORKDIR /app/packages/sgrna_scorer
RUN pip install -e .

WORKDIR /app/packages/pt_guide_design
RUN pip install -e .

WORKDIR /app

# Copy the service code
COPY server/services/guide_design/service.py .

CMD ["python", "service.py"]