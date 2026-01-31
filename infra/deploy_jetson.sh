#!/bin/bash
# Deploy OpenClawdIRL to a Jetson device via SSH
# Usage: ./infra/deploy_jetson.sh <user@host> [project_dir]
#
# This script copies the project to the Jetson and sets up dependencies.
# It will NOT touch any existing files outside the project directory.

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <user@host> [project_dir]"
    echo "Example: $0 user@192.168.1.100 ~/OpenClawdIRL"
    exit 1
fi

SSH_TARGET="$1"
REMOTE_DIR="${2:-~/OpenClawdIRL}"

echo "=== Deploying OpenClawdIRL to $SSH_TARGET:$REMOTE_DIR ==="

# Create remote directory
echo "[1/4] Creating remote directory..."
ssh "$SSH_TARGET" "mkdir -p $REMOTE_DIR"

# Sync project files (excluding node_modules, venv, etc.)
echo "[2/4] Syncing project files..."
rsync -avz --progress \
    --exclude '.git' \
    --exclude 'node_modules' \
    --exclude '.next' \
    --exclude '__pycache__' \
    --exclude 'venv' \
    --exclude '.venv' \
    --exclude '.env' \
    --exclude 'dump.rdb' \
    --exclude '.DS_Store' \
    ./ "$SSH_TARGET:$REMOTE_DIR/"

# Install Python dependencies on device
echo "[3/4] Installing Python dependencies on device..."
ssh "$SSH_TARGET" "cd $REMOTE_DIR && pip3 install -r requirements.txt"

echo "[4/4] Done!"
echo ""
echo "=== Next steps on the Jetson ==="
echo "1. SSH into the device: ssh $SSH_TARGET"
echo "2. cd $REMOTE_DIR"
echo "3. Copy .env.example to .env and fill in your keys:"
echo "   cp .env.example .env && nano .env"
echo "4. Start Redis (must have redis-stack installed):"
echo "   make redis"
echo "5. Setup Redis indexes (once):"
echo "   make redis-setup"
echo "6. Start the backend API:"
echo "   make backend"
echo "7. Start the perception pipeline:"
echo "   make perception"
echo "8. (On laptop) Start the dashboard:"
echo "   cd dashboard && npm install && npm run dev"
