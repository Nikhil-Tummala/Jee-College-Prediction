# Makefile for JEE College Prediction project

.PHONY: help install train predict test clean lint format setup-dev

# Default target
help:
	@echo "Available targets:"
	@echo "  help        - Show this help message"
	@echo "  install     - Install dependencies"
	@echo "  setup-dev   - Set up development environment"
	@echo "  train       - Train the model"
	@echo "  predict     - Make a prediction (requires RANK, GENDER, SEAT_TYPE)"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo "  clean       - Clean up generated files"
	@echo "  docs        - Generate documentation"

# Install dependencies
install:
	pip install -r requirements.txt

# Set up development environment
setup-dev:
	pip install -r requirements.txt
	pip install -e .
	mkdir -p logs
	mkdir -p data/raw
	mkdir -p data/processed
	mkdir -p models

# Train the model
train:
	python main.py train

# Make a prediction (usage: make predict RANK=1000 GENDER=Male SEAT_TYPE=Open)
predict:
	python main.py predict $(RANK) $(GENDER) $(SEAT_TYPE)

# Run tests
test:
	python -m pytest tests/ -v

# Run tests with coverage
test-coverage:
	python -m pytest tests/ --cov=src --cov-report=html

# Run linting
lint:
	flake8 src/ tests/
	mypy src/

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Clean up generated files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/

# Generate documentation
docs:
	cd docs && make html

# Run the complete pipeline
pipeline: clean install train

# Development shortcuts
dev-install: setup-dev

# Example usage
example:
	@echo "Training model..."
	python main.py train
	@echo "Making prediction..."
	python main.py predict 1000 Male Open

# Check code quality
quality: lint test

# Build distribution
build:
	python setup.py sdist bdist_wheel

# Install in development mode
dev:
	pip install -e .

# Run Jupyter notebook
notebook:
	jupyter notebook notebooks/

# Install pre-commit hooks
pre-commit:
	pre-commit install

# Update dependencies
update-deps:
	pip install --upgrade -r requirements.txt

# Backup data
backup:
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz data/ models/ config/

# Show project structure
tree:
	tree -I '__pycache__|*.pyc|.git|venv|env'
