# Auto-Deploy Setup for Pt_sgRNA_Prediction

This guide explains how to set up automatic deployment for Pt_sgRNA_Prediction, matching the setup used by BuckEuchre.

## Overview

The auto-deploy system automatically pulls the latest code from GitHub and redeploys the application whenever changes are pushed to the `main` branch. This matches the deployment workflow used by BuckEuchre.

## Prerequisites

1. **GitHub SSH Key**: A deploy key configured for your repository
2. **Server Access**: SSH access to your production server
3. **GitHub Webhook** (optional): For automatic triggering on push events

## Setup Instructions

### Step 1: Create GitHub Deploy Key

If you don't already have a deploy key set up (shared with BuckEuchre):

```bash
# On your production server, generate a new SSH key for deployments
ssh-keygen -t ed25519 -f ~/.ssh/github_deploy -N ""

# Display the public key
cat ~/.ssh/github_deploy.pub
```

### Step 2: Add Deploy Key to GitHub

1. Go to your GitHub repository: `Settings` → `Deploy keys`
2. Click `Add deploy key`
3. Paste the public key from Step 1
4. Give it a title (e.g., "Production Server Deploy Key")
5. Check `Allow write access` if you need to push tags or commits
6. Click `Add key`

**Note**: If BuckEuchre already uses `~/.ssh/github_deploy`, you can reuse the same key for both repositories.

### Step 3: Configure Repository Path

The `deploy.sh` script assumes the repository is cloned to `~/Pt_sgRNA_Prediction/`. If your path is different, update the script:

```bash
# Edit the deploy script
nano ~/Pt_sgRNA_Prediction/server/scripts/deploy.sh

# Update this line to match your actual path:
cd ~/Pt_sgRNA_Prediction/server/
```

### Step 4: Test the Deploy Script

```bash
# Make sure the script is executable
chmod +x ~/Pt_sgRNA_Prediction/server/scripts/deploy.sh

# Test it manually
~/Pt_sgRNA_Prediction/server/scripts/deploy.sh
```

## Auto-Deploy Options

### Option A: GitHub Actions (Recommended)

Create `.github/workflows/deploy.yml` in your repository:

```yaml
name: Deploy to Production

on:
  push:
    branches:
      - main

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to server
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.SSH_HOST }}
          username: ${{ secrets.SSH_USER }}
          key: ${{ secrets.SSH_PRIVATE_KEY }}
          script: |
            cd ~/Pt_sgRNA_Prediction/server/scripts
            ./deploy.sh
```

**Required GitHub Secrets:**
- `SSH_HOST`: Your server's IP address or hostname
- `SSH_USER`: SSH username (e.g., `ubuntu`, `deploy`)
- `SSH_PRIVATE_KEY`: Private SSH key for server access

### Option B: GitHub Webhook + Webhook Receiver

1. **Install webhook receiver** (e.g., `webhook` from adnanh/webhook):

```bash
# On your server
sudo apt-get install webhook

# Create webhook config
sudo nano /etc/webhook.conf
```

Add configuration:
```json
[
  {
    "id": "pt-sgrna-deploy",
    "execute-command": "/home/USERNAME/Pt_sgRNA_Prediction/server/scripts/deploy.sh",
    "command-working-directory": "/home/USERNAME/Pt_sgRNA_Prediction/server/scripts",
    "pass-arguments-to-command": [],
    "trigger-rule": {
      "match": {
        "type": "payload-hmac-sha256",
        "secret": "YOUR_WEBHOOK_SECRET",
        "parameter": {
          "source": "header",
          "name": "X-Hub-Signature-256"
        }
      }
    }
  }
]
```

2. **Start webhook service**:
```bash
sudo systemctl start webhook
sudo systemctl enable webhook
```

3. **Configure GitHub Webhook**:
   - Go to repository: `Settings` → `Webhooks` → `Add webhook`
   - Payload URL: `http://YOUR_SERVER_IP:9000/hooks/pt-sgrna-deploy`
   - Content type: `application/json`
   - Secret: `YOUR_WEBHOOK_SECRET` (same as in config)
   - Events: `Just the push event`
   - Active: ✓

### Option C: Cron Job (Polling)

Set up a cron job to periodically check for updates:

```bash
# Edit crontab
crontab -e

# Add this line to check every 5 minutes
*/5 * * * * cd ~/Pt_sgRNA_Prediction && git fetch && [ $(git rev-parse HEAD) != $(git rev-parse origin/main) ] && ~/Pt_sgRNA_Prediction/server/scripts/deploy.sh >> /var/log/pt-sgrna-deploy.log 2>&1
```

**Note**: This is less efficient than webhooks but doesn't require external services.

## Manual Deployment

You can also run the deploy script manually:

```bash
cd ~/Pt_sgRNA_Prediction/server/scripts
./deploy.sh
```

Or use the production start script:

```bash
cd ~/Pt_sgRNA_Prediction/server/scripts
./production-start.sh
```

## Deployment Process

When `deploy.sh` runs, it:

1. **Fetches latest code** from GitHub (`git fetch --all --prune`)
2. **Resets to main branch** (`git reset --hard origin/main`)
3. **Builds Docker images** (`docker compose build --pull`)
4. **Stops existing containers** (`docker compose down`)
5. **Starts new containers** (`docker compose up -d`)
6. **Waits for health checks** (checks db and redis are healthy)
7. **Logs completion** with timestamp and commit hash

## Troubleshooting

### Deploy Script Fails with SSH Error

```bash
# Test SSH connection to GitHub
ssh -i ~/.ssh/github_deploy -T git@github.com

# If it fails, check:
# 1. Key is added to GitHub deploy keys
# 2. Key permissions: chmod 600 ~/.ssh/github_deploy
# 3. SSH config allows the key
```

### Services Don't Start

```bash
# Check logs
cd ~/Pt_sgRNA_Prediction/server
docker-compose logs

# Check service status
docker-compose ps

# Verify environment variables
cat .env
```

### Port Conflicts

If you see port conflicts, verify:
- Port 3001 is available (Pt_sgRNA_Prediction app)
- Port 5433 is available (Pt_sgRNA_Prediction postgres)
- Port 6379 is available (Redis)

```bash
# Check what's using ports
sudo lsof -i :3001
sudo lsof -i :5433
sudo lsof -i :6379
```

### Health Checks Fail

The deploy script waits for services to be healthy. If they don't become healthy:

```bash
# Check individual service health
docker-compose ps

# View service logs
docker-compose logs db
docker-compose logs redis
docker-compose logs app
```

## Monitoring Deployments

### View Deployment Logs

If using cron or webhook, check logs:

```bash
# Cron logs
tail -f /var/log/pt-sgrna-deploy.log

# Webhook logs
sudo journalctl -u webhook -f

# Docker logs
cd ~/Pt_sgRNA_Prediction/server
docker-compose logs -f
```

### Deployment Notifications

You can add notification hooks to `deploy.sh`:

```bash
# Add to end of deploy.sh
# Send notification (example with curl)
curl -X POST https://hooks.slack.com/services/YOUR/WEBHOOK/URL \
  -d '{"text":"Pt_sgRNA_Prediction deployed: $(git rev-parse --short HEAD)"}'
```

## Security Considerations

1. **SSH Key Security**:
   - Keep `~/.ssh/github_deploy` private (chmod 600)
   - Use a dedicated deploy key (not your personal SSH key)
   - Rotate keys periodically

2. **Webhook Security**:
   - Always use HMAC signature verification
   - Use strong, random secrets
   - Consider IP whitelisting

3. **Script Permissions**:
   - Ensure only authorized users can execute deploy scripts
   - Review script contents before running

## Comparison with BuckEuchre

Both applications now use the same deployment pattern:

| Feature | BuckEuchre | Pt_sgRNA_Prediction |
|---------|-----------|---------------------|
| Deploy Script | `~/buckEuchre/scripts/deploy.sh` | `~/Pt_sgRNA_Prediction/server/scripts/deploy.sh` |
| SSH Key | `~/.ssh/github_deploy` | `~/.ssh/github_deploy` (shared) |
| Docker Compose | `docker-compose.yml` | `docker-compose.yml` |
| Health Checks | Backend + Postgres | DB + Redis |

## Next Steps

1. ✅ Set up GitHub deploy key
2. ✅ Configure auto-deploy (choose Option A, B, or C above)
3. ✅ Test deployment manually
4. ✅ Monitor first automatic deployment
5. ✅ Set up deployment notifications (optional)

## Support

For issues or questions:
- Check deployment logs
- Review Docker Compose status
- Verify environment variables
- Check GitHub Actions/webhook logs (if applicable)

