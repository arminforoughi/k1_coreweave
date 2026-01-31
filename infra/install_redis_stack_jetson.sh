#!/bin/bash
# Install Redis Stack on Jetson (ARM64/aarch64)
# Redis Stack includes RediSearch (for vector search) and RedisJSON
# This script installs from source since Redis Stack binaries aren't available for ARM64.
#
# Run this ONCE on the Jetson device.

set -e

echo "=== Installing Redis Stack dependencies for Jetson (ARM64) ==="

# Install Redis server if not present
if ! command -v redis-server &> /dev/null; then
    echo "Installing Redis server..."
    sudo apt-get update
    sudo apt-get install -y redis-server
fi

# We'll use a Docker approach since compiling RediSearch from source is complex
echo ""
echo "Option 1: Use Docker (Recommended if Docker is available)"
echo "  docker run -d --name redis-stack -p 6379:6379 redis/redis-stack-server:latest"
echo ""
echo "Option 2: Use redis-server with RedisJSON and RediSearch modules"
echo "  You'll need to compile these from source for ARM64."
echo "  See: https://github.com/RediSearch/RediSearch"
echo "  See: https://github.com/RedisJSON/RedisJSON"
echo ""
echo "Option 3: Use Redis Cloud (no local install needed)"
echo "  Sign up at: https://redis.io/try-free"
echo "  Use the REDIS_URL in your .env file"
echo ""

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "Docker is available. Starting Redis Stack container..."
    docker pull redis/redis-stack-server:latest 2>/dev/null || true
    docker run -d --name redis-stack -p 6379:6379 --restart unless-stopped redis/redis-stack-server:latest 2>/dev/null || echo "Container may already exist. Try: docker start redis-stack"
    echo "Redis Stack should be running on port 6379"
else
    echo "Docker not found. Please install Docker or use Redis Cloud."
    echo "To install Docker on Jetson:"
    echo "  sudo apt-get install -y docker.io"
    echo "  sudo usermod -aG docker \$USER"
fi
