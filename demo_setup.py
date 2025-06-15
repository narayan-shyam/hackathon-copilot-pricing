#!/usr/bin/env python3
"""
Demo script to showcase the Production ML Project Setup Tool
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run a demo of the project setup tool"""
    
    print("🎯 Production ML Project Setup Tool Demo")
    print("=" * 50)
    
    # Show available options
    print("\n📋 Available Setup Options:")
    print("1. Basic setup (default)")
    print("2. Custom project name")
    print("3. Production environment")
    print("4. With Azure Key Vault")
    print("5. Validation only")
    print("6. Show help")
    
    choice = input("\nSelect option (1-6): ").strip()
    
    base_cmd = [sys.executable, "project_setup.py"]
    
    if choice == "1":
        cmd = base_cmd
        print("\n🚀 Running basic setup...")
        
    elif choice == "2":
        project_name = input("Enter project name: ").strip() or "my-ml-project"
        cmd = base_cmd + ["--project-name", project_name]
        print(f"\n🚀 Running setup with project name: {project_name}")
        
    elif choice == "3":
        cmd = base_cmd + ["--environment", "production"]
        print("\n🚀 Running production environment setup...")
        
    elif choice == "4":
        vault_url = input("Enter Azure Key Vault URL: ").strip()
        if vault_url:
            cmd = base_cmd + ["--azure-vault-url", vault_url]
        else:
            cmd = base_cmd
        print("\n🚀 Running setup with Azure Key Vault integration...")
        
    elif choice == "5":
        cmd = base_cmd + ["--validate-only"]
        print("\n🔍 Running validation check...")
        
    elif choice == "6":
        cmd = base_cmd + ["--help"]
        print("\n📖 Showing help...")
        
    else:
        print("❌ Invalid choice. Running basic setup...")
        cmd = base_cmd
    
    # Add verbose flag for demo
    if choice != "6":
        cmd.append("--verbose")
    
    try:
        print(f"\n⚡ Executing: {' '.join(cmd)}")
        print("-" * 50)
        
        result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            print("\n✅ Setup completed successfully!")
            print("\n📁 Files created in current directory:")
            
            # List created files/directories
            current_dir = Path.cwd()
            for item in sorted(current_dir.iterdir()):
                if item.name not in ['.git', '__pycache__', '.pytest_cache']:
                    icon = "📁" if item.is_dir() else "📄"
                    print(f"  {icon} {item.name}")
                    
        else:
            print(f"\n⚠️  Setup completed with warnings (exit code: {result.returncode})")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running setup: {e}")

if __name__ == "__main__":
    main()
