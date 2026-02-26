#!/bin/bash

# ==============================================================================
# Pt_sgRNA_Prediction - Production Startup Script
# ==============================================================================
# 
# This script starts all production services using Docker Compose
# 
# Prerequisites:
# 1. Docker and Docker Compose installed
# 2. .env file created with all required variables
# 
# Usage:
#   ./production-start.sh
# 
# ==============================================================================

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Pt_sgRNA_Prediction - Production Setup${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Docker is running"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${RED}❌ .env not found!${NC}"
    echo ""
    echo "Please create .env file with all required environment variables:"
    echo "  POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB"
    echo "  SES_HOST, SES_PORT, SES_USER, SES_PASS, SES_FROM"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓${NC} .env found"
echo ""

# Source environment variables
export $(cat .env | grep -v '^#' | xargs)

# Verify critical environment variables
missing_vars=()

if [ -z "$POSTGRES_USER" ]; then
    missing_vars+=("POSTGRES_USER")
fi

if [ -z "$POSTGRES_PASSWORD" ]; then
    missing_vars+=("POSTGRES_PASSWORD")
fi

if [ -z "$POSTGRES_DB" ]; then
    missing_vars+=("POSTGRES_DB")
fi

if [ ${#missing_vars[@]} -ne 0 ]; then
    echo -e "${RED}❌ Missing environment variables:${NC}"
    for var in "${missing_vars[@]}"; do
        echo -e "${RED}  - $var${NC}"
    done
    echo ""
    echo "Please update .env with all required values."
    echo ""
    exit 1
fi

echo -e "${GREEN}✓${NC} Environment variables validated"
echo ""

# Pull latest images
echo -e "${GREEN}Pulling latest Docker images...${NC}"
docker-compose pull

echo ""

# Build custom images
echo -e "${GREEN}Building application images...${NC}"
docker-compose build --no-cache

echo ""

# Start services
echo -e "${GREEN}Starting production services...${NC}"
docker-compose up -d

echo ""

# Wait for services to be healthy
echo -e "${GREEN}Waiting for services to be healthy...${NC}"
echo ""

max_attempts=60
attempt=0

while [ $attempt -lt $max_attempts ]; do
    healthy_count=$(docker-compose ps | grep -c "healthy" || true)
    
    # We expect at least db and redis to be healthy
    if [ "$healthy_count" -ge 2 ]; then
        echo -e "${GREEN}✓${NC} All services are healthy!"
        break
    fi
    
    attempt=$((attempt + 1))
    if [ $attempt -eq $max_attempts ]; then
        echo -e "${RED}❌ Services failed to start within ${max_attempts} seconds${NC}"
        echo ""
        echo "Check logs with:"
        echo "  docker-compose logs"
        echo ""
        docker-compose ps
        exit 1
    fi
    
    sleep 1
    echo -n "."
done

echo ""
echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Production Services Started!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "Services:"
docker-compose ps
echo ""
echo "Access your application at:"
echo "  http://localhost:3001 (direct access)"
echo "  or via your configured domain through BuckEuchre nginx"
echo ""
echo "Useful commands:"
echo "  View logs:        docker-compose logs -f"
echo "  Stop services:    docker-compose down"
echo "  Restart:          docker-compose restart"
echo "  View status:      docker-compose ps"
echo ""
echo "⚠️  Remember to:"
echo "  - Configure domain routing in BuckEuchre nginx (nginx/conf.d/sgrna.conf)"
echo "  - Set up database backups"
echo "  - Configure monitoring and alerting"
echo "  - Review security settings"
echo ""

