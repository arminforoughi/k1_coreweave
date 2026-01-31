#!/bin/bash
# ============================================================================
# OpenClawdIRL Device Recon Script
# ============================================================================
# This script is READ-ONLY. It does not install, modify, or create anything.
# Run this on the Jetson BEFORE deploying to check if resources are sufficient.
#
# Usage (from your laptop):
#   ssh user@jetson-ip 'bash -s' < infra/recon_device.sh
#
# Or copy it over and run directly:
#   scp infra/recon_device.sh user@jetson-ip:/tmp/
#   ssh user@jetson-ip 'bash /tmp/recon_device.sh'
# ============================================================================

set -e

echo "============================================"
echo "  OpenClawdIRL Device Recon"
echo "  $(date)"
echo "============================================"
echo ""

# --- System Info ---
echo "=== SYSTEM INFO ==="
echo "Hostname: $(hostname)"
echo "Kernel:   $(uname -r)"
echo "Arch:     $(uname -m)"
if [ -f /etc/nv_tegra_release ]; then
    echo "Tegra:    $(cat /etc/nv_tegra_release | head -1)"
fi
if command -v jetson_release &> /dev/null; then
    echo "JetPack:  $(jetson_release 2>/dev/null | head -3 || echo 'N/A')"
fi
echo ""

# --- Disk Space ---
echo "=== DISK SPACE ==="
df -h / | tail -1 | awk '{print "Root (/):  " $4 " available of " $2 " (" $5 " used)"}'
df -h ~ 2>/dev/null | tail -1 | awk '{print "Home (~):  " $4 " available of " $2 " (" $5 " used)"}' || true
echo ""
echo "We need approximately 3-5 GB free for:"
echo "  - Python venv + dependencies (~2 GB)"
echo "  - YOLO model weights (~20 MB)"
echo "  - MobileNetV2 weights (~14 MB)"
echo "  - Redis data (~100 MB)"
echo "  - Docker image for Redis Stack (~500 MB)"
echo ""

# --- Memory ---
echo "=== MEMORY ==="
free -h | head -2
echo ""
echo "Swap:"
free -h | grep -i swap || echo "No swap"
echo ""

# --- GPU / CUDA ---
echo "=== GPU / CUDA ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi available but query failed"
elif command -v tegrastats &> /dev/null; then
    echo "tegrastats available (Jetson GPU detected)"
    echo "(Run 'tegrastats --interval 1000' for live stats)"
else
    echo "No nvidia-smi or tegrastats found"
fi

if [ -d /usr/local/cuda ]; then
    echo "CUDA found: $(ls /usr/local/cuda/version* 2>/dev/null | head -1 || echo '/usr/local/cuda exists')"
    echo "nvcc: $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not in PATH')"
else
    echo "No /usr/local/cuda directory found"
fi
echo ""

# --- Python ---
echo "=== PYTHON ==="
for py in python3 python; do
    if command -v $py &> /dev/null; then
        echo "$py: $($py --version 2>&1) at $(which $py)"
    fi
done
echo ""
echo "pip:"
pip3 --version 2>/dev/null || pip --version 2>/dev/null || echo "pip not found"
echo ""
echo "Key packages already installed:"
pip3 list 2>/dev/null | grep -iE "torch|opencv|numpy|redis|ultralytics|fastapi|weave|wandb" || echo "  (could not check)"
echo ""

# --- Docker ---
echo "=== DOCKER ==="
if command -v docker &> /dev/null; then
    echo "Docker: $(docker --version)"
    echo "Running containers:"
    docker ps --format "  {{.Names}}\t{{.Image}}\t{{.Ports}}" 2>/dev/null || echo "  Cannot list (permission issue? Try: sudo usermod -aG docker \$USER)"
else
    echo "Docker not installed"
    echo "We need Docker for Redis Stack. Install with: sudo apt-get install -y docker.io"
fi
echo ""

# --- Camera ---
echo "=== CAMERA ==="
if ls /dev/video* 2>/dev/null; then
    echo "Video devices found:"
    ls -la /dev/video* 2>/dev/null
else
    echo "No /dev/video* devices found"
fi
echo ""
if command -v v4l2-ctl &> /dev/null; then
    echo "v4l2-ctl available. Camera details:"
    v4l2-ctl --list-devices 2>/dev/null || echo "  (no devices or permission denied)"
else
    echo "v4l2-ctl not available (install v4l-utils for camera details)"
fi
echo ""

# --- Ports ---
echo "=== PORT CHECK ==="
echo "Checking if our required ports are free..."
for port in 6379 8000 3000; do
    if ss -tlnp 2>/dev/null | grep -q ":${port} " || netstat -tlnp 2>/dev/null | grep -q ":${port} "; then
        echo "  Port $port: IN USE (conflict!)"
        ss -tlnp 2>/dev/null | grep ":${port} " || netstat -tlnp 2>/dev/null | grep ":${port} "
    else
        echo "  Port $port: FREE"
    fi
done
echo ""

# --- Existing project directory ---
echo "=== PROJECT DIRECTORY ==="
if [ -d ~/OpenClawdIRL ]; then
    echo "WARNING: ~/OpenClawdIRL already exists!"
    echo "Contents:"
    ls -la ~/OpenClawdIRL/ 2>/dev/null | head -20
else
    echo "~/OpenClawdIRL does not exist (clean slate)"
fi
echo ""

# --- Network ---
echo "=== NETWORK ==="
echo "IP addresses:"
ip -4 addr show 2>/dev/null | grep inet | grep -v 127.0.0.1 | awk '{print "  " $2}' || hostname -I 2>/dev/null || echo "  Could not determine"
echo ""
echo "Internet connectivity:"
ping -c 1 -W 2 8.8.8.8 &>/dev/null && echo "  Internet: OK" || echo "  Internet: UNREACHABLE"
ping -c 1 -W 2 pypi.org &>/dev/null && echo "  PyPI: OK" || echo "  PyPI: UNREACHABLE (may need proxy)"
echo ""

# --- Summary ---
echo "============================================"
echo "  RECON COMPLETE"
echo "============================================"
echo ""
echo "Review the output above and confirm:"
echo "  1. At least 3-5 GB disk space free"
echo "  2. At least 4 GB RAM available (8 GB+ preferred)"
echo "  3. CUDA / GPU is accessible"
echo "  4. Python 3.8+ is installed"
echo "  5. Docker is available (for Redis Stack)"
echo "  6. Camera device exists"
echo "  7. Ports 6379, 8000 are free"
echo "  8. Internet works (for pip install)"
echo ""
echo "If all checks pass, proceed with deployment."
