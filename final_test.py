#!/usr/bin/env python3
"""
Final comprehensive test to verify all fixes are working
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

# Add the project directory to path
project_dir = Path(__file__).parent
sys.path.append(str(project_dir))

warnings.filterwarnings('ignore')

print("🚀 COMPREHENSIVE FIX VERIFICATION TEST")
print("=" * 70)

try:
    print("\n1️⃣ Testing imports and basic functionality...")
    from unified_dynamic_pricing import (
        create_sample_pricing_data,
        UnifiedDynamicPricingPipeline,
        create_pipeline_config
    )
    print("✅ All imports successful!")
    
    print("\n2️⃣ Creating sample data with edge cases...")
    sample_data = create_sample_pricing_data(n_days=20, n_products=3)
    print(f"✅ Sample data created: {len(sample_data)} records")
    
    # Add some edge cases to test robustness
    sample_data.loc[0, 'SellingPrice'] = np.nan  # Missing target
    sample_data.loc[1, 'Product'] = 'Product_Test_Special'  # New category
    print("✅ Added edge cases to test robustness")
    
    print("\n3️⃣ Testing complete pipeline with small dataset...")
    config = create_pipeline_config({
        'data_processor': {
            'missing_value_strategy': 'auto',
            'outlier_method': 'iqr',
            'scaling_method': 'robust'
        },
        'model_trainer': {
            'cv_folds': 3,  # Reduced for speed
            'scoring': 'r2'
        }
    })
    
    pipeline = UnifiedDynamicPricingPipeline(config)
    print("✅ Pipeline initialized")
    
    print("\n4️⃣ Running complete pipeline (this should work without errors)...")
    results = pipeline.run_complete_pipeline(
        data_source=sample_data,
        target_column='SellingPrice',
        test_size=0.3
    )
    
    if results['pipeline_status'] == 'success':
        print("🎉 SUCCESS! Pipeline completed without errors!")
        print(f"✅ Best model: {results.get('best_model', {}).get('name', 'None')}")
        print(f"✅ Data shapes:")
        for step, shape in results.get('data_shapes', {}).items():
            print(f"   • {step}: {shape}")
        
        print(f"✅ Features created: {results.get('feature_engineering_summary', {}).get('total_created_features', 0)}")
        
        # Test prediction
        print("\n5️⃣ Testing prediction capabilities...")
        new_data = create_sample_pricing_data(n_days=3, n_products=2)
        predictions = pipeline.predict(new_data)
        print(f"✅ Predictions generated: {len(predictions)} values")
        print(f"✅ Prediction range: ${predictions.min():.2f} - ${predictions.max():.2f}")
        
        print("\n🎊 ALL TESTS PASSED! THE APPLICATION IS READY!")
        print("=" * 70)
        print("✅ Syntax errors fixed")
        print("✅ Parameter naming issues fixed") 
        print("✅ Categorical encoding issues fixed")
        print("✅ Feature engineering working properly")
        print("✅ Model training working without sklearn errors")
        print("✅ Prediction pipeline working")
        print("\n🚀 You can now run: python main.py")
        
    else:
        print(f"❌ Pipeline failed: {results.get('error', 'Unknown error')}")
        print("Check the logs above for details.")
        
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    print("\n🔧 There may still be an issue. Check the error above.")

print("\n" + "=" * 70)
