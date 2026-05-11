.PHONY: help install train serve test docker-build docker-up docker-down docker-logs clean

help:
	@echo "Targets:"
	@echo "  install       - install python deps"
	@echo "  train         - run local LoRA sanity-check training"
	@echo "  serve         - run FastAPI locally (no docker)"
	@echo "  test          - run pytest"
	@echo "  docker-build  - build docker image"
	@echo "  docker-up     - start docker stack (api + redis)"
	@echo "  docker-down   - stop docker stack"
	@echo "  docker-logs   - tail api logs"
	@echo "  clean         - remove caches"

install:
	uv pip install -r requirements.txt

train:
	python training/train_lora.py

serve:
	uvicorn serving.api.main:app --reload --port 8000

test:
	.venv/bin/pytest tests/ -v

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f api

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache
