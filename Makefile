.PHONY: install dev test lint run diagram

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

# Example: make run LOOP=react TASK="What is 128 * 47?"
run:
	python -m claude_agents.cli $(LOOP) "$(TASK)" --trace

diagram:
	python3 scripts/generate_arch_diagram.py assets/architecture.json --outdir assets
