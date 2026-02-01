#!/bin/bash
# ============================================================================
# OpenClawdIRL — Thin Client Deploy to Jetson
# ============================================================================
# WHAT THIS DOES:
#   Copies ONLY the Jetson client (camera + YOLO + HTTP POST) to the device.
#   Everything else (embedding, KNN, Redis, research, dashboard) runs on laptop.
#
# ISOLATION GUARANTEES:
#   - Only creates/modifies files inside $REMOTE_DIR (default: ~/OpenClawdIRL)
#   - Uses a Python venv INSIDE the project dir (no global pip install)
#   - No sudo commands. No system-level modifications.
#   - No modifications to .bashrc, .profile, or any dotfiles.
#   - Complete cleanup: rm -rf ~/OpenClawdIRL
#
# WHAT GOES TO THE JETSON (tiny footprint):
#   - perception/jetson_client.py        (the thin client)
#   - perception/requirements-jetson.txt (ultralytics, opencv, requests)
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
echo "  OpenClawdIRL Thin Client Deploy"
echo "============================================"
echo "Target:    $SSH_TARGET"
echo "Dir:       $REMOTE_DIR"
echo ""
echo "ONLY deploying: jetson_client.py + minimal deps"
echo "NO other files or services will be on the device."
echo "============================================"
echo ""

# Confirm
read -p "Proceed? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

# Step 1: Create remote directory
echo ""
echo "[1/4] Creating project directory..."
ssh "$SSH_TARGET" "mkdir -p $REMOTE_DIR/perception"

# Step 2: Copy only the thin client files
echo "[2/4] Copying thin client files..."
scp perception/jetson_client.py "$SSH_TARGET:$REMOTE_DIR/perception/"
scp perception/requirements-jetson.txt "$SSH_TARGET:$REMOTE_DIR/perception/"

# Step 3: Create venv and install deps
echo "[3/4] Setting up Python venv + deps on device..."
ssh "$SSH_TARGET" "cd $REMOTE_DIR && python3 -m venv $REMOTE_DIR/venv && $REMOTE_DIR/venv/bin/pip install --upgrade pip && $REMOTE_DIR/venv/bin/pip install -r $REMOTE_DIR/perception/requirements-jetson.txt"

# Step 4: Create a convenience run script
echo "[4/4] Creating run script..."
ssh "$SSH_TARGET" "cat > $REMOTE_DIR/run.sh << 'SCRIPT'
#!/bin/bash
# Usage: ./run.sh <laptop-ip> [camera-index] [fps]
BACKEND=\"http://\${1:-localhost}:8000\"
CAMERA=\"\${2:-0}\"
FPS=\"\${3:-3}\"
cd \$(dirname \$0)
./venv/bin/python perception/jetson_client.py --backend \$BACKEND --camera \$CAMERA --fps \$FPS
SCRIPT
chmod +x $REMOTE_DIR/run.sh"

echo ""
echo "============================================"
echo "  DEPLOY COMPLETE"
echo "============================================"
echo ""
echo "On the Jetson:"
echo "  ssh $SSH_TARGET"
echo "  cd $REMOTE_DIR"
echo "  ./run.sh <your-laptop-ip>    # Start sending frames to laptop"
echo ""
echo "On your laptop (first):"
echo "  make redis && make redis-setup   # Start Redis (once)"
echo "  make backend                     # Start API server"
echo "  make dashboard                   # Start dashboard"
echo ""
echo "Cleanup: rm -rf $REMOTE_DIR"
