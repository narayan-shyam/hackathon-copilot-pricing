"""
Project Structure Creation with Production-Ready Templates
"""

from pathlib import Path
from typing import Dict, Any
import logging

from .exceptions import DirectoryCreationError
from .utilities import DataValidator, retry

logger = logging.getLogger(__name__)


class ProjectStructureManager:
    """Manages creation of production-ready ML project structure with all necessary components"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.validator = DataValidator()
    
    def create_project_structure(self):
        """Create comprehensive ML project directory structure with all templates"""
        
        # Define complete project structure
        project_structure = {
            "src": {
                "data": {"__init__.py": ""},
                "models": {"__init__.py": ""},
                "utils": {"__init__.py": ""},
                "__init__.py": ""
            },
            "config": {},
            "data": {"raw": {}, "processed": {}, "external": {}, "interim": {}},
            "models": {"trained": {}, "artifacts": {}},
            "notebooks": {"exploratory": {}, "experiments": {}},
            "tests": {
                "unit": {"__init__.py": ""},
                "integration": {"__init__.py": ""},
                "__init__.py": ""
            },
            "scripts": {},
            "docs": {"inputs": {}, "api": {}, "user_guide": {}},
            "logs": {},
            "requirements.txt": self._get_requirements_template(),
            "setup.py": self._get_setup_template(),
            "Dockerfile": self._get_dockerfile_template(),
            "docker-compose.yml": self._get_docker_compose_template(),
            ".gitignore": self._get_gitignore_template(),
            "README.md": self._get_readme_template(),
            "Makefile": self._get_makefile_template()
        }
        
        # Create the complete structure
        self._create_structure_recursive(self.base_path, project_structure)
        logger.info(f"Project structure created successfully at: {self.base_path}")
    
    @retry(max_attempts=3, delay=0.5)
    def _create_structure_recursive(self, current_path: Path, structure: Dict[str, Any]):
        """Recursively create directory structure with error handling"""
        try:
            current_path.mkdir(parents=True, exist_ok=True)
            
            for name, content in structure.items():
                item_path = current_path / name
                
                if isinstance(content, dict):
                    # It's a directory
                    self._create_structure_recursive(item_path, content)
                else:
                    # It's a file
                    with open(item_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
        except Exception as e:
            logger.error(f"Failed to create structure at {current_path}: {str(e)}")
            raise DirectoryCreationError(f"Directory creation failed: {str(e)}")
    
    def _get_requirements_template(self) -> str:
        """Get requirements.txt template"""
        return '''# Core ML libraries
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
# MLflow (lightweight version to avoid Windows path issues)
mlflow-skinny>=2.0.0

# Data processing
PyYAML>=6.0
joblib>=1.3.0

# Azure integration
azure-keyvault-secrets>=4.7.0
azure-identity>=1.13.0
azure-monitor-opentelemetry>=1.0.0

# Development tools
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0

# API and web
fastapi>=0.100.0
uvicorn>=0.23.0
requests>=2.31.0

# Utilities
click>=8.1.0
tqdm>=4.65.0
python-dotenv>=1.0.0
'''
    
    def _get_setup_template(self) -> str:
        """Get setup.py template"""
        return '''from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ml-pricing-project",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="ML Pricing Prediction Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
'''
    
    def _get_dockerfile_template(self) -> str:
        """Get Dockerfile template"""
        return '''FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
    
    def _get_docker_compose_template(self) -> str:
        """Get docker-compose.yml template"""
        return '''version: '3.8'

services:
  ml-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=development
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow
      - postgres
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - ./models:/app/models

  mlflow:
    image: python:3.10-slim
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:mlflow@postgres:5432/mlflow
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
    volumes:
      - mlflow_artifacts:/mlflow/artifacts
    command: >
      bash -c "pip install mlflow psycopg2-binary &&
               mlflow server --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow 
               --default-artifact-root /mlflow/artifacts 
               --host 0.0.0.0 --port 5000"
    depends_on:
      - postgres

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=mlflow
      - POSTGRES_USER=mlflow
      - POSTGRES_PASSWORD=mlflow
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
  mlflow_artifacts:
'''
    
    def _get_gitignore_template(self) -> str:
        """Get .gitignore template"""
        return '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
models/trained/*
models/artifacts/*
!models/trained/.gitkeep
!models/artifacts/.gitkeep

# Logs
logs/*.log

# MLflow
mlruns/
mlflow.db
mlflow_*.db

# Environment variables
.env
.env.local
.env.*.local

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# pytest
.pytest_cache/
.coverage
htmlcov/

# mypy
.mypy_cache/
.dmypy.json
dmypy.json
'''
    
    def _get_readme_template(self) -> str:
        """Get README.md template"""
        return '''# ML Pricing Project

## Overview
Production-ready machine learning project for pricing prediction with comprehensive MLOps practices.

## Features
- ✅ Structured logging with JSON formatting
- ✅ Azure Key Vault integration for secure configuration
- ✅ MLflow experiment tracking
- ✅ Comprehensive error handling with retry mechanisms
- ✅ Rate limiting and data validation utilities
- ✅ Circuit breaker patterns for fault tolerance
- ✅ Production-ready Docker setup
- ✅ Comprehensive testing framework

## Setup

### Prerequisites
- Python 3.8+
- Docker (optional)
- Azure account (for Key Vault integration)

### Installation
```bash
# Clone repository
git clone <repository-url>
cd ml-pricing-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\\\Scripts\\\\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training
```bash
python scripts/train_model.py --data-path data/raw/training_data.csv
```

### Docker
```bash
# Build and run with Docker Compose
docker-compose up --build

# Access MLflow UI
open http://localhost:5000
```

## Testing
```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## License
MIT License - see LICENSE file for details.
'''
    
    def _get_makefile_template(self) -> str:
        """Get Makefile template"""
        return '''SHELL := /bin/bash
.PHONY: help install test lint format clean docker-build docker-up docker-down

help: ## Show this help message
\t@echo 'Usage: make [target]'
\t@echo ''
\t@echo 'Targets:'
\t@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \\033[36m%-15s\\033[0m %s\\\\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
\tpip install -r requirements.txt

test: ## Run tests
\tpytest tests/

lint: ## Lint code
\tflake8 src/ tests/

format: ## Format code
\tblack src/ tests/

clean: ## Clean up generated files
\tfind . -type f -name "*.pyc" -delete
\tfind . -type d -name "__pycache__" -delete

docker-build: ## Build Docker images
\tdocker-compose build

docker-up: ## Start Docker services
\tdocker-compose up -d

docker-down: ## Stop Docker services
\tdocker-compose down

setup-project: ## Initialize project structure
\tpython project_setup.py
'''
