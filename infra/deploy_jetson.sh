#!/bin/bash
# ============================================================================
# OpenClawdIRL — Safe Deploy to Jetson
# ============================================================================
# ISOLATION GUARANTEES:
#   - Only creates/modifies files inside $REMOTE_DIR (default: ~/OpenClawdIRL)
#   - Uses a Python venv INSIDE the project dir (no global pip install)
#   - Redis runs as a Docker container (no host system changes)
#   - No sudo commands. No system-level modifications.
#   - No modifications to .bashrc, .profile, or any dotfiles.
#   - Complete cleanup: rm -rf ~/OpenClawdIRL && docker rm redis-stack
#
# Usage: ./infra/deploy_jetson.sh <user@host> [project_dir]
# ============================================================================

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <user@host> [project_dir]"
    echo "Example: $0 user@192.168.1.100 ~/OpenClawdIRL"
    exit 1
fi

SSH_TARGET="$1"
REMOTE_DIR="${2:-~/OpenClawdIRL}"

echo "============================================"
echo "  OpenClawdIRL Safe Deploy"
echo "============================================"
echo "Target:   $SSH_TARGET"
echo "Dir:      $REMOTE_DIR"
echo ""
echo "ISOLATION: All changes confined to $REMOTE_DIR"
echo "           No system files will be modified."
echo "============================================"
echo ""

# Confirm before proceeding
read -p "Proceed with deployment? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

# Step 1: Create remote directory (the ONLY directory we create)
echo ""
echo "[1/5] Creating project directory on device..."
ssh "$SSH_TARGET" "mkdir -p $REMOTE_DIR"

# Step 2: Sync project files
echo "[2/5] Syncing project files to $REMOTE_DIR..."
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
    --exclude 'TEAMMATE_REVIEW.md' \
    ./ "$SSH_TARGET:$REMOTE_DIR/"

# Step 3: Create isolated Python venv INSIDE the project directory
echo "[3/5] Creating Python virtual environment inside project dir..."
ssh "$SSH_TARGET" "cd $REMOTE_DIR && python3 -m venv $REMOTE_DIR/venv"

# Step 4: Install dependencies into the venv (NOT globally)
echo "[4/5] Installing Python dependencies into project venv..."
ssh "$SSH_TARGET" "cd $REMOTE_DIR && $REMOTE_DIR/venv/bin/pip install --upgrade pip && $REMOTE_DIR/venv/bin/pip install -r $REMOTE_DIR/requirements.txt"

# Step 5: Copy .env template if .env doesn't exist
echo "[5/5] Setting up environment file..."
ssh "$SSH_TARGET" "cd $REMOTE_DIR && if [ ! -f .env ]; then cp .env.example .env; echo 'Created .env from template — edit it with your API keys'; else echo '.env already exists, not overwriting'; fi"

echo ""
echo "============================================"
echo "  DEPLOY COMPLETE"
echo "============================================"
echo ""
echo "Next steps (on the Jetson via SSH):"
echo ""
echo "  ssh $SSH_TARGET"
echo "  cd $REMOTE_DIR"
echo ""
echo "  # 1. Edit .env with your WANDB_API_KEY:"
echo "  nano .env"
echo ""
echo "  # 2. Start Redis Stack (Docker, isolated container):"
echo "  docker run -d --name redis-stack -p 6379:6379 --restart unless-stopped redis/redis-stack-server:latest"
echo ""
echo "  # 3. Setup Redis indexes (one time):"
echo "  $REMOTE_DIR/venv/bin/python infra/setup_redis.py"
echo ""
echo "  # 4. Start the backend API:"
echo "  $REMOTE_DIR/venv/bin/python -m backend.api"
echo ""
echo "  # 5. Start the perception pipeline:"
echo "  $REMOTE_DIR/venv/bin/python -m perception.pipeline"
echo ""
echo "  # 6. On your LAPTOP, start the dashboard:"
echo "  cd dashboard && NEXT_PUBLIC_API_URL=http://<jetson-ip>:8000 npm run dev"
echo ""
echo "To completely remove everything from the device:"
echo "  docker stop redis-stack && docker rm redis-stack"
echo "  rm -rf $REMOTE_DIR"
