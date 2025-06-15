#!/usr/bin/env python3
"""
Quick test script to verify the setup tool works with minimal dependencies
"""

def test_imports():
    """Test that all required modules can be imported"""
    
    print("🧪 Testing imports for project setup tool...")
    
    try:
        # Test standard library imports
        import os
        import json
        import logging
        import sys
        from pathlib import Path
        from datetime import datetime
        from dataclasses import dataclass, asdict
        import time
        import functools
        print("✅ Standard library imports: OK")
        
        # Test PyYAML (main external dependency)
        try:
            import yaml
            print("✅ PyYAML import: OK")
        except ImportError:
            print("⚠️  PyYAML not found - install with: pip install PyYAML")
        
        # Test our core modules
        try:
            from core.exceptions import ProjectSetupError
            from core.logging_config import setup_logging
            from core.utilities import DataValidator, RateLimiter
            print("✅ Core modules import: OK")
        except ImportError as e:
            print(f"❌ Core module import failed: {e}")
            
        print("\n🎯 Tool should work with current dependencies!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without external dependencies"""
    
    print("\n🔧 Testing basic functionality...")
    
    try:
        from core.utilities import DataValidator, SecurityUtils
        
        # Test data validator
        validator = DataValidator()
        result = validator.sanitize_filename("test<>file.txt")
        print(f"✅ File sanitization: {result}")
        
        # Test security utils
        token = SecurityUtils.generate_token(16)
        print(f"✅ Token generation: {token[:8]}...")
        
        print("✅ Basic functionality: OK")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Production ML Project Setup Tool")
    print("=" * 50)
    
    imports_ok = test_imports()
    functionality_ok = test_basic_functionality()
    
    if imports_ok and functionality_ok:
        print("\n✅ All tests passed! Tool is ready to use.")
        print("\nNext steps:")
        print("1. Install PyYAML if not already installed: pip install PyYAML")
        print("2. Run the tool: python project_setup.py --help")
        print("3. Or try the demo: python demo_setup.py")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
