#!/usr/bin/env bash
set -Eeuo pipefail

# Navigate to the server directory where docker-compose.yml is located
cd ~/Pt_sgRNA_Prediction/server/
export GIT_SSH_COMMAND='ssh -i ~/.ssh/github_deploy_sgrna -o IdentitiesOnly=yes'

echo "[deploy] Updating repo…"
git fetch --all --prune
git reset --hard origin/main

echo "[deploy] Building & starting…"
# Use explicit production compose file (ignore override file for local dev)
docker compose -f docker-compose.yml build --pull
docker compose -f docker-compose.yml down --remove-orphans || true
docker compose -f docker-compose.yml up -d --remove-orphans

echo "[deploy] Waiting for services to be healthy…"
sleep 10
max_attempts=60
attempt=0
while [ $attempt -lt $max_attempts ]; do
    # Check for app, db, and redis health
    if docker compose -f docker-compose.yml ps | grep -q "healthy.*db" && \
       docker compose -f docker-compose.yml ps | grep -q "healthy.*redis"; then
        echo "[deploy] Services are healthy!"
        break
    fi
    attempt=$((attempt + 1))
    if [ $attempt -eq $max_attempts ]; then
        echo "[deploy] ⚠️  Services may not be fully healthy yet. Check with: docker compose ps"
    fi
    sleep 1
done

echo "[deploy] Done @ $(date). Commit: $(git rev-parse --short HEAD)"

