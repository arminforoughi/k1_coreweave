.PHONY: setup redis-setup perception backend dashboard all clean

# Install Python dependencies
setup:
	pip install -r requirements.txt

# Setup Redis indexes and consumer groups
redis-setup:
	python infra/setup_redis.py

# Run the perception pipeline
perception:
	python -m perception.pipeline

# Run the backend API
backend:
	python -m backend.api

# Install and run the dashboard
dashboard:
	cd dashboard && npm install && npm run dev

# Run Redis locally (assumes redis-server is installed)
redis:
	redis-server --loadmodule /opt/redis-stack/lib/rejson.so --loadmodule /opt/redis-stack/lib/redisearch.so

# Run everything (in separate terminals)
all:
	@echo "Run these in separate terminals:"
	@echo "  make redis        # Start Redis (if local)"
	@echo "  make redis-setup  # Create indexes (once)"
	@echo "  make backend      # Start API server"
	@echo "  make perception   # Start camera pipeline"
	@echo "  make dashboard    # Start web dashboard"

# Clean up
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
