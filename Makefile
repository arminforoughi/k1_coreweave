.PHONY: setup redis-setup backend dashboard jetson-client jetson-ros all clean

# ===== LAPTOP (runs everything except camera) =====

# Install Python dependencies (laptop)
setup:
	pip install -r requirements.txt

# Setup Redis indexes and consumer groups
redis-setup:
	python infra/setup_redis.py

# Run the backend API (receives crops from Jetson, does embedding + KNN)
backend:
	python -m backend.api

# Install and run the dashboard
dashboard:
	cd dashboard && npm install && npm run dev

# Run Redis via Docker (laptop)
redis:
	docker run -d --name redis-stack -p 6379:6379 --restart unless-stopped redis/redis-stack-server:latest

# ===== JETSON (thin client only) =====

# Run the Jetson client (camera + YOLO + POST to backend)
# Usage: make jetson-client BACKEND=http://<laptop-ip>:8000
jetson-client:
	python perception/jetson_client.py --backend $(BACKEND)

# Run the Jetson client with ROS 2 camera topic + MiDaS depth
# Usage: make jetson-ros BACKEND=http://<laptop-ip>:8000
jetson-ros:
	python perception/jetson_client.py --backend $(BACKEND) --ros

# Install Jetson-only dependencies
jetson-setup:
	pip install -r perception/requirements-jetson.txt

# ===== GENERAL =====

# Run everything (instructions)
all:
	@echo "=== Thin Device Architecture ==="
	@echo ""
	@echo "ON YOUR LAPTOP (run each in a separate terminal):"
	@echo "  make redis          # Start Redis Stack (once)"
	@echo "  make redis-setup    # Create indexes (once)"
	@echo "  make backend        # Start API + embedder + KNN"
	@echo "  make dashboard      # Start web dashboard"
	@echo ""
	@echo "ON THE JETSON:"
	@echo "  make jetson-setup                                    # Install deps (once)"
	@echo "  make jetson-ros BACKEND=http://<laptop-ip>:8000      # ROS 2 + depth (primary)"
	@echo "  make jetson-client BACKEND=http://<laptop-ip>:8000   # Fallback (no ROS 2)"

# Clean up
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
