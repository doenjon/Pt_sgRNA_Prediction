#!/bin/bash

# Get current git commit
export GIT_COMMIT=$(git rev-parse --short HEAD)

# Enable BuildKit
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Build with BuildKit enabled
docker-compose build --parallel

echo "Build completed for git commit: $GIT_COMMIT" 