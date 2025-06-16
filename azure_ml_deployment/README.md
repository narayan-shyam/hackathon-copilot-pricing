# Azure ML Deployment for Dynamic Pricing Pipeline

## 📁 Module 3: Model Deployment Automation

This directory contains all the necessary files for deploying the trained dynamic pricing model from Databricks to Azure ML Managed Endpoints.

## 🗂️ Files Overview

### Core Deployment Files
- **`deploy_model.py`** - Main deployment script for Azure ML
- **`score.py`** - Scoring script for Azure ML endpoint
- **`test_api.py`** - Comprehensive API testing suite

### Configuration Files
- **`.env.template`** - Environment variables template
- **`requirements.txt`** - Python dependencies

## 🚀 Quick Start Guide

### 1. Prerequisites
```bash
# Install required packages
pip install azure-ai-ml azure-identity requests

# Set up environment variables
cp ../.env.template .env
# Edit .env with your Azure credentials
```

### 2. Environment Variables
Update your `.env` file with:
```bash
AZURE_ML_SUBSCRIPTION_ID=your-subscription-id
AZURE_ML_RESOURCE_GROUP=your-resource-group
AZURE_ML_WORKSPACE_NAME=your-workspace-name
SCORING_URI=https://your-endpoint.region.inference.ml.azure.com/score
API_KEY=your-endpoint-key
```

### 3. Deploy Model to Azure ML
```bash
python deploy_model.py
```

### 4. Test the Deployed API
```bash
python test_api.py
```

## 📊 Expected Workflow

### Step 1: Model Registration
The deployment script will:
1. Connect to your Azure ML workspace
2. Register the model from Databricks MLflow
3. Create model metadata and versioning

### Step 2: Endpoint Creation
1. Create Azure ML Managed Online Endpoint
2. Configure authentication and scaling
3. Set up routing and traffic management

### Step 3: Model Deployment
1. Deploy the model to the endpoint
2. Configure compute resources
3. Set environment variables and dependencies

### Step 4: Testing & Validation
1. Run comprehensive API tests
2. Validate response format and performance
3. Generate test reports

## 🧪 API Testing Features

The testing suite includes:

### Basic Functionality Tests
- ✅ Single prediction requests
- ✅ Batch prediction requests
- ✅ Response format validation
- ✅ Error handling validation

### Business Scenario Tests
- 🏷️ High demand pricing
- 📦 Low inventory scenarios
- 🏆 Competitive pricing
- 🎯 Weekend premium pricing

### Performance Tests
- ⚡ Load testing with multiple requests
- 📈 Response time monitoring
- 🎯 Success rate tracking
- 📊 Throughput measurement

### Error Handling Tests
- ❌ Missing required fields
- 🚫 Invalid data types
- ⚠️ Edge cases and boundary conditions

## 📄 Sample API Request

```json
{
  "base_price": 100.0,
  "cost": 60.0,
  "competitor_price": 95.0,
  "demand": 150,
  "inventory_level": 500,
  "customer_engagement": 0.75,
  "market_demand_factor": 1.2,
  "seasonal_factor": 1.1,
  "price_to_cost_ratio": 1.67,
  "profit_margin": 0.4,
  "is_profitable": true,
  "price_vs_competitor": 1.05,
  "price_change": 2.0,
  "demand_change": 10,
  "price_elasticity": -0.5,
  "demand_trend_7d": 145.0,
  "price_volatility_7d": 5.2,
  "avg_price_30d": 98.5,
  "inventory_velocity": 0.3,
  "competitive_position": 1.05,
  "revenue_per_unit": 100.0,
  "profit_per_unit": 40.0,
  "day_of_week": 3,
  "month": 6,
  "quarter": 2,
  "is_weekend": 0,
  "is_month_end": 0,
  "category_avg_price": 102.0,
  "category_price_rank": 5,
  "price_vs_category_avg": 0.98,
  "demand_supply_ratio": 0.3,
  "profit_optimization_score": 12.0,
  "market_positioning_score": 1.27
}
```

## 📊 Sample API Response

```json
{
  "predicted_price": 103.45,
  "confidence_interval": [98.28, 108.62],
  "input_features": {
    "base_price": 100.0,
    "competitor_price": 95.0,
    "demand": 150,
    "inventory_level": 500
  },
  "business_metrics": {
    "profit_margin": 41.98,
    "competitive_position": 1.089
  },
  "model_metadata": {
    "model_name": "dynamic_pricing_model",
    "model_version": "1",
    "features_used": 29,
    "prediction_timestamp": "2025-06-16T10:30:45.123456"
  }
}
```

## 🔧 Troubleshooting

### Common Issues

#### Authentication Errors
```bash
# Ensure Azure CLI is logged in
az login

# Or use environment variables for service principal
export AZURE_CLIENT_ID=your-client-id
export AZURE_CLIENT_SECRET=your-client-secret
export AZURE_TENANT_ID=your-tenant-id
```

#### Model Registration Failures
- Verify MLflow model URI from Databricks
- Check Azure ML workspace permissions
- Ensure model artifacts are accessible

#### Endpoint Deployment Failures
- Check compute quota in Azure ML
- Verify resource group permissions
- Review deployment logs in Azure portal

#### API Testing Failures
- Verify endpoint URL and API key
- Check feature schema compatibility
- Review Azure ML endpoint logs

### Debug Commands

```bash
# Check Azure ML workspace connection
az ml workspace show -n your-workspace -g your-resource-group

# List registered models
az ml model list

# Check endpoint status
az ml online-endpoint show -n dynamic-pricing-endpoint

# View deployment logs
az ml online-deployment get-logs -n blue -e dynamic-pricing-endpoint
```

## 📈 Performance Expectations

### Response Times
- **Single Prediction**: < 2 seconds
- **Batch Prediction (5 items)**: < 5 seconds
- **Cold Start**: < 10 seconds

### Throughput
- **Expected RPS**: 10-50 requests/second
- **Peak Load**: 100+ requests/second
- **Concurrent Users**: 20+

### Availability
- **Target Uptime**: 99.9%
- **Auto-scaling**: Enabled
- **Health Monitoring**: Active

## 🔄 Next Steps

After successful deployment:

1. **Module 4**: Implement comprehensive testing framework
2. **Module 5**: Set up monitoring and logging infrastructure
3. **Module 6**: Build automated retraining pipeline
4. **Module 7**: Create CI/CD pipeline
5. **Module 8**: Develop web application frontend

## 📞 Support

For issues with deployment:
1. Check Azure ML workspace logs
2. Review endpoint diagnostics
3. Verify model compatibility
4. Test with sample data first

## 🎯 Success Criteria

✅ Model successfully registered in Azure ML  
✅ Endpoint created and accessible  
✅ API responds to test requests  
✅ Performance meets requirements  
✅ Error handling works correctly  
✅ Ready for production use  

---

**🚀 Ready to deploy your dynamic pricing model to Azure ML!**
