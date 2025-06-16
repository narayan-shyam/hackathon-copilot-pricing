# 📦 Installation Guide for Dynamic Pricing Pipeline

## 🎯 **Quick Setup Options**

### **Option 1: Complete Installation (Recommended)**
```bash
# Clone or extract project
cd hackathon-copilot-pricing

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install all dependencies
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

### **Option 2: Azure Deployment Only**
```bash
# For Azure ML deployment only
cd azure_ml_deployment
pip install -r requirements-minimal.txt
```

### **Option 3: Local Development Only**
```bash
# Core ML dependencies without Azure services
pip install pandas numpy scikit-learn mlflow matplotlib jupyter
```

## 🔧 **Installation by Use Case**

### **📊 Data Science & Local Development**
```bash
pip install pandas>=2.0.0 numpy>=1.24.0 scikit-learn>=1.3.0 mlflow>=2.8.0 matplotlib>=3.7.0 jupyter>=1.0.0 xgboost lightgbm
```

### **☁️ Azure Cloud Deployment**
```bash
pip install azure-ai-ml>=1.12.0 azure-identity>=1.15.0 pandas>=2.0.0 scikit-learn>=1.3.0 mlflow>=2.8.0 requests>=2.31.0
```

### **🧱 Databricks Environment**
```bash
pip install databricks-sql-connector>=2.5.0 mlflow>=2.8.0 pandas>=2.0.0 scikit-learn>=1.3.0
```

### **🌐 Web API Development**
```bash
pip install fastapi>=0.100.0 uvicorn[standard]>=0.23.0 pandas>=2.0.0 scikit-learn>=1.3.0 pydantic>=2.5.0
```

## 🐳 **Docker Installation**

### **Using Docker Compose**
```bash
docker-compose up --build
```

### **Manual Docker Build**
```bash
docker build -t dynamic-pricing-pipeline .
docker run -p 8000:8000 dynamic-pricing-pipeline
```

## 🚀 **Installation Verification**

### **Test Core Functionality**
```python
# Test basic imports
import pandas as pd
import numpy as np
import sklearn
import mlflow

# Test pipeline
from unified_dynamic_pricing import UnifiedDynamicPricingPipeline
pipeline = UnifiedDynamicPricingPipeline()
print("✅ Core installation successful!")
```

### **Test Azure Integration**
```python
# Test Azure imports
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
print("✅ Azure ML SDK installed successfully!")
```

### **Run Quick Test**
```bash
# Full functionality test
python -c "from unified_dynamic_pricing import create_sample_pricing_data; print('✅ Pipeline ready!')"
```

## 🔧 **Troubleshooting**

### **Common Installation Issues**

#### **1. Azure ML SDK Conflicts**
```bash
# If you get conflicts between azure-ai-ml and azureml-core
pip uninstall azureml-core azureml-train-automl-client
pip install azure-ai-ml>=1.12.0
```

#### **2. MLflow UI Issues**
```bash
# If MLflow UI doesn't work, use minimal version
pip uninstall mlflow
pip install mlflow-skinny>=2.8.0
```

#### **3. XGBoost/LightGBM Installation Failures**
```bash
# For Windows users with build issues
pip install xgboost lightgbm --only-binary=all

# Or use conda
conda install xgboost lightgbm
```

#### **4. Databricks Connector Issues**
```bash
# If databricks-sql-connector fails
pip install --upgrade pip setuptools wheel
pip install databricks-sql-connector>=2.5.0
```

#### **5. Memory Issues During Installation**
```bash
# Install with limited parallel builds
pip install --no-cache-dir -r requirements.txt
```

## 🎯 **Platform-Specific Instructions**

### **Windows**
```bash
# Use Windows Subsystem for Linux (WSL) for best experience
wsl --install

# Or use Anaconda/Miniconda
conda create -n pricing-pipeline python=3.10
conda activate pricing-pipeline
pip install -r requirements.txt
```

### **macOS**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.10

# Create environment and install
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### **Linux (Ubuntu/Debian)**
```bash
# Install Python and dependencies
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev build-essential

# Create environment and install
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🏢 **Enterprise/Production Setup**

### **Using Poetry (Recommended for Production)**
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Create pyproject.toml and install
poetry init
poetry add pandas numpy scikit-learn mlflow azure-ai-ml
poetry install
```

### **Using Pipenv**
```bash
# Install Pipenv
pip install pipenv

# Install dependencies
pipenv install -r requirements.txt
pipenv shell
```

### **Using Conda (Data Science Teams)**
```bash
# Create conda environment
conda create -n pricing-pipeline python=3.10
conda activate pricing-pipeline

# Install from conda-forge when possible
conda install -c conda-forge pandas numpy scikit-learn
pip install azure-ai-ml mlflow
```

## 📋 **Dependency Summary**

| Category | Essential | Optional | Notes |
|----------|-----------|----------|-------|
| **Core ML** | pandas, numpy, scikit-learn | xgboost, lightgbm | Always needed |
| **Experiment Tracking** | mlflow | mlflow-skinny | MLflow UI optional |
| **Azure Cloud** | azure-ai-ml, azure-identity | azure-storage-* | Only for cloud deployment |
| **Databricks** | databricks-sql-connector | pyspark | Only for Databricks |
| **API** | requests | fastapi, uvicorn | For deployment/testing |
| **Development** | pytest | black, flake8 | For development only |
| **Visualization** | matplotlib | seaborn, plotly | For analysis/reports |

## ✅ **Installation Success Checklist**

After installation, verify:

- [ ] Python version 3.8+ installed
- [ ] Virtual environment activated
- [ ] Core dependencies installed (pandas, numpy, scikit-learn)
- [ ] MLflow available (`mlflow --version`)
- [ ] Azure SDK available (if needed)
- [ ] Pipeline imports work
- [ ] Sample data generation works
- [ ] Basic model training works

## 🎉 **Ready to Go!**

Once installation is complete:

1. **Test the pipeline**: Run `python -m unified_dynamic_pricing`
2. **Check documentation**: Review `CONSOLIDATED_DOCUMENTATION.md`
3. **Try Azure deployment**: Follow `azure_ml_deployment/README.md`
4. **Start developing**: Begin with your own data!

Your dynamic pricing pipeline is now ready for action! 🚀
