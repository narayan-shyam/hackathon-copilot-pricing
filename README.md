# ML Pricing Project

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
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

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
