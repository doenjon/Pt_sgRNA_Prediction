FROM python:3.9-slim

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
COPY services/guide_design/requirements.txt .

# Install redis package first
RUN pip install --no-cache-dir redis

# Install packages in smaller chunks
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir numpy pandas biopython gffutils pybedtools tqdm pyfaidx

# Install scikit-learn separately
RUN pip install --no-cache-dir scikit-learn==1.2.2

# Install tensorflow separately
RUN pip install --no-cache-dir tensorflow==2.12.0

# Create package directories and install packages in development mode
COPY ../pt_guide_design /app/packages/pt_guide_design/
COPY ../sgrna_scorer /app/packages/sgrna_scorer/

# Install the packages in development mode
RUN cd /app/packages/sgrna_scorer && pip install -e . && \
    cd /app/packages/pt_guide_design && pip install -e .

# Copy the service code
COPY services/guide_design/service.py .

CMD ["python", "service.py"]