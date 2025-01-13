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
COPY services/guide_design/requirements.txt .

# Install packages in smaller chunks
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir numpy pandas biopython gffutils pybedtools tqdm pyfaidx

# Install scikit-learn separately
RUN pip install --no-cache-dir scikit-learn==1.2.2

# Install tensorflow separately
RUN pip install --no-cache-dir tensorflow==2.12.0

# Create package directories
RUN mkdir -p /app/packages/sgrna_scorer /app/packages/pt_guide_design

# The actual packages will be mounted as volumes
# Copy the service code
COPY services/guide_design/service.py .

CMD ["python", "service.py"]