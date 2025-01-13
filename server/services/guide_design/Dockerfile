FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    bwa \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the packages
COPY sgrna_scorer /app/packages/sgrna_scorer
COPY pt_guide_design /app/packages/pt_guide_design

# Install sgrna_scorer first since pt_guide_design depends on it
WORKDIR /app/packages/sgrna_scorer
RUN pip install -e .

WORKDIR /app/packages/pt_guide_design
RUN pip install -e .

WORKDIR /app

# Copy the service code
COPY services/guide_design/service.py .

CMD ["python", "service.py"] 