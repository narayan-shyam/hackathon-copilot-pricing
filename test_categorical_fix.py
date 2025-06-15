#!/usr/bin/env python3
"""
Test script to verify categorical encoding fixes
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add the project directory to path
project_dir = Path(__file__).parent
sys.path.append(str(project_dir))

print("🧪 TESTING CATEGORICAL ENCODING FIXES")
print("=" * 60)

try:
    print("\n1️⃣ Testing imports...")
    from unified_dynamic_pricing import (
        create_sample_pricing_data,
        UnifiedDynamicPricingPipeline,
        create_pipeline_config
    )
    print("✅ All imports successful!")
    
    print("\n2️⃣ Creating sample data...")
    sample_data = create_sample_pricing_data(n_days=10, n_products=3)
    print(f"✅ Sample data created: {len(sample_data)} records")
    print(f"✅ Data types before processing:")
    print(sample_data.dtypes)
    print(f"\n✅ Sample data preview:")
    print(sample_data.head(3))
    
    print("\n3️⃣ Testing data processing pipeline...")
    config = create_pipeline_config()
    pipeline = UnifiedDynamicPricingPipeline(config)
    
    # Test data processing step by step
    print("\n📊 Step 1: Load and validate data")
    df = pipeline.load_data(sample_data)
    validation_results = pipeline.validate_data(df)
    print(f"✅ Data validation: {validation_results['schema_valid']}")
    
    print("\n📊 Step 2: Preprocess data")
    df_processed = pipeline.preprocess_data(df)
    print(f"✅ Data processed. Shape: {df.shape} -> {df_processed.shape}")
    print(f"✅ Processed data types:")
    print(df_processed.dtypes)
    
    # Check for categorical columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    print(f"✅ Remaining categorical columns: {categorical_cols}")
    
    # Check for encoded columns
    encoded_cols = [col for col in df_processed.columns if col.endswith('_encoded')]
    print(f"✅ Encoded categorical columns: {encoded_cols}")
    
    print("\n📊 Step 3: Feature engineering")
    df_features = pipeline.engineer_features(df_processed)
    print(f"✅ Feature engineering completed. Shape: {df_processed.shape} -> {df_features.shape}")
    
    print("\n📊 Step 4: Prepare training data")
    X_train, X_test, y_train, y_test = pipeline.prepare_training_data(df_features, 'SellingPrice', 0.2)
    print(f"✅ Training data prepared. Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"✅ Training data types:")
    print(X_train.dtypes)
    
    # Final check: ensure no categorical columns in training data
    final_categorical = X_train.select_dtypes(include=['object']).columns.tolist()
    if final_categorical:
        print(f"❌ ERROR: Still have categorical columns in training data: {final_categorical}")
        print("This would cause the sklearn error!")
    else:
        print("✅ SUCCESS: No categorical columns in training data!")
    
    print("\n4️⃣ Testing a quick model training...")
    # Test with a simple model to verify the fix
    from sklearn.linear_model import LinearRegression
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"✅ Model training successful! Made {len(predictions)} predictions")
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ Categorical encoding issue has been fixed!")
    print("✅ The application should now run without sklearn errors!")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    print("\n🔧 There may still be an issue. Check the error above.")
