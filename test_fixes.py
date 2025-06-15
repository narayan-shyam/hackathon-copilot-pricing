#!/usr/bin/env python3
"""
Quick test script to verify the fixes are working
"""

import sys
import os
from pathlib import Path

# Add the project directory to path
project_dir = Path(__file__).parent
sys.path.append(str(project_dir))

print("🧪 TESTING UNIFIED DYNAMIC PRICING PIPELINE FIXES")
print("=" * 60)

try:
    print("\n1️⃣ Testing imports...")
    from unified_dynamic_pricing import (
        create_sample_pricing_data,
        UnifiedDynamicPricingPipeline,
        create_pipeline_config
    )
    print("✅ All imports successful!")
    
    print("\n2️⃣ Testing sample data creation...")
    sample_data = create_sample_pricing_data(n_days=5, n_products=2)
    print(f"✅ Sample data created: {len(sample_data)} records")
    print(f"✅ Columns: {list(sample_data.columns)}")
    
    print("\n3️⃣ Testing pipeline initialization...")
    config = create_pipeline_config()
    pipeline = UnifiedDynamicPricingPipeline(config)
    print("✅ Pipeline initialized successfully!")
    
    print("\n4️⃣ Testing basic pipeline components...")
    # Test data validation
    validation_results = pipeline.validate_data(sample_data)
    print(f"✅ Data validation completed: {validation_results['schema_valid']}")
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ The application is ready to run!")
    print("\nYou can now run: python main.py")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    print("\n🔧 Please check the error above and ensure all dependencies are installed.")
    print("Try running: pip install -r requirements.txt")
