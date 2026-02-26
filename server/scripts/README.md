# Pt_sgRNA_Prediction - Deployment Scripts

This directory contains scripts for deploying and managing the Pt_sgRNA_Prediction application in production.

## Scripts

### `deploy.sh`
**Purpose**: Automated deployment script that pulls latest code and redeploys the application.

**Usage**:
```bash
./deploy.sh
```

**What it does**:
- Fetches latest code from GitHub
- Resets to `origin/main`
- Builds and starts Docker containers
- Waits for services to be healthy

**Used by**: GitHub Actions, webhooks, or cron jobs for auto-deployment.

### `production-start.sh`
**Purpose**: Manual production startup script with environment validation.

**Usage**:
```bash
./production-start.sh
```

**What it does**:
- Validates Docker is running
- Checks for `.env` file
- Validates environment variables
- Pulls latest images
- Builds and starts services
- Waits for health checks

**Used for**: Initial setup or manual deployments.

### `backup-database.sh`
**Purpose**: Creates a backup of the PostgreSQL database.

**Usage**:
```bash
./backup-database.sh
```

**What it does**:
- Creates a timestamped SQL backup
- Stores backups in `../backups/` directory
- Automatically cleans up backups older than 7 days

**Used for**: Regular database backups (can be scheduled with cron).

## Setup

See [AUTO_DEPLOY_SETUP.md](./AUTO_DEPLOY_SETUP.md) for detailed instructions on setting up automatic deployments.

## Requirements

- Docker and Docker Compose installed
- `.env` file configured in `../` directory
- GitHub deploy key set up (for auto-deploy)
- SSH access to production server

## Related Documentation

- [AUTO_DEPLOY_SETUP.md](./AUTO_DEPLOY_SETUP.md) - Auto-deploy configuration guide
- [../docker-compose.yml](../docker-compose.yml) - Docker Compose configuration
- [../../MULTI_APP_CONFIGURATION.md](../../MULTI_APP_CONFIGURATION.md) - Multi-app setup guide

