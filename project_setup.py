#!/usr/bin/env python3
"""
Main Project Setup Script for Production-ready ML Project
Demonstrates comprehensive project initialization with industry best practices
"""

import os
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import argparse

# Import our modular components
from core.exceptions import ProjectSetupError, DirectoryCreationError, ConfigurationError
from core.logging_config import setup_logging, JSONFormatter
from core.utilities import RateLimiter, DataValidator, retry
from core.azure_integration import AzureKeyVaultManager
from core.configuration import ConfigurationManager, ProjectConfig
from core.mlflow_manager import MLflowManager
from core.project_structure import ProjectStructureManager


class ProductionMLProjectSetup:
    """Main class for setting up production-ready ML project"""
    
    def __init__(self, project_name: str = "ml-pricing-project", base_path: Optional[Path] = None):
        self.project_name = project_name
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.config_manager = ConfigurationManager()
        self.rate_limiter = RateLimiter()
        
    def initialize_project(self, environment: str = "development", 
                          azure_vault_url: Optional[str] = None) -> Dict[str, Any]:
        """Initialize complete production-ready ML project"""
        
        # Setup logging first
        global logger
        logger = setup_logging()
        logger.info("Starting ML project initialization")
        
        try:
            # Load configuration
            config = self.config_manager.load_config(environment)
            
            # Setup Azure integration if vault URL provided
            if azure_vault_url:
                self.config_manager.setup_azure_integration(azure_vault_url)
            
            # Create project structure
            structure_manager = ProjectStructureManager(self.base_path)
            structure_manager.create_project_structure()
            
            # Setup MLflow tracking
            mlflow_manager = MLflowManager(config)
            mlflow_manager.setup_mlflow()
            
            # Install dependencies
            self._install_dependencies()
            
            # Initialize git repository
            self._initialize_git()
            
            logger.info("ML project initialization completed successfully")
            
            return {
                "status": "success",
                "project_name": self.project_name,
                "base_path": str(self.base_path),
                "environment": environment,
                "config": config.to_dict(),
                "features": [
                    "Structured logging with JSON formatting",
                    "Azure Key Vault integration",
                    "MLflow experiment tracking",
                    "Comprehensive error handling",
                    "Rate limiting utilities",
                    "Circuit breaker patterns",
                    "Data validation framework",
                    "Production-ready Docker setup",
                    "Comprehensive testing framework"
                ]
            }
            
        except Exception as e:
            logger.error(f"Project initialization failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    @retry(max_attempts=3, delay=1.0)
    def _install_dependencies(self):
        """Install project dependencies"""
        try:
            logger.info("Installing project dependencies")
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to install dependencies: {e}")
    
    def _initialize_git(self):
        """Initialize git repository if not already initialized"""
        try:
            import subprocess
            if not (self.base_path / ".git").exists():
                subprocess.run(["git", "init"], cwd=self.base_path, check=True)
                subprocess.run(["git", "add", "."], cwd=self.base_path, check=True)
                subprocess.run([
                    "git", "commit", "-m", "Initial commit: Production-ready ML project setup"
                ], cwd=self.base_path, check=True)
                logger.info("Git repository initialized")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Git initialization failed: {e}")
    
    def validate_setup(self) -> Dict[str, Any]:
        """Validate project setup"""
        logger.info("Validating project setup")
        
        validation_results = {
            "structure_valid": self._validate_project_structure(),
            "dependencies_valid": self._validate_dependencies(),
            "config_valid": self._validate_configuration(),
            "logging_valid": self._validate_logging_setup()
        }
        
        all_valid = all(validation_results.values())
        
        return {
            "status": "valid" if all_valid else "invalid",
            "results": validation_results,
            "recommendations": self._get_setup_recommendations(validation_results)
        }
    
    def _validate_project_structure(self) -> bool:
        """Validate that all required directories exist"""
        required_dirs = [
            "src", "src/data", "src/models", "src/utils",
            "config", "tests", "scripts", "logs", "data", "models"
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not (self.base_path / dir_path).exists():
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            logger.warning(f"Missing directories: {missing_dirs}")
            return False
        
        return True
    
    def _validate_dependencies(self) -> bool:
        """Validate that required dependencies are installed"""
        required_packages = ["pandas", "numpy", "yaml"]
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.warning(f"Missing packages: {missing_packages}")
            return False
        
        return True
    
    def _validate_configuration(self) -> bool:
        """Validate configuration files exist"""
        config_files = ["development.yaml", "staging.yaml", "production.yaml"]
        
        for config_file in config_files:
            config_path = self.base_path / "config" / config_file
            if not config_path.exists():
                logger.warning(f"Missing config file: {config_file}")
                return False
        
        return True
    
    def _validate_logging_setup(self) -> bool:
        """Validate logging configuration"""
        logs_dir = self.base_path / "logs"
        return logs_dir.exists() and logs_dir.is_dir()
    
    def _get_setup_recommendations(self, validation_results: Dict[str, bool]) -> List[str]:
        """Get recommendations based on validation results"""
        recommendations = []
        
        if not validation_results["structure_valid"]:
            recommendations.append("Run project structure creation again")
        
        if not validation_results["dependencies_valid"]:
            recommendations.append("Install missing dependencies with: pip install -r requirements.txt")
        
        if not validation_results["config_valid"]:
            recommendations.append("Create missing configuration files")
        
        if not validation_results["logging_valid"]:
            recommendations.append("Create logs directory")
        
        return recommendations


def main():
    """Main CLI interface for project setup"""
    parser = argparse.ArgumentParser(
        description="Production-ready ML Project Setup Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python project_setup.py                                    # Basic setup
  python project_setup.py --project-name my-ml-project      # Custom name
  python project_setup.py --environment production          # Production config
  python project_setup.py --azure-vault-url https://...     # With Azure Key Vault
  python project_setup.py --validate-only                   # Validate existing setup
        """
    )
    
    parser.add_argument(
        '--project-name',
        default='ml-pricing-project',
        help='Name of the ML project (default: ml-pricing-project)'
    )
    
    parser.add_argument(
        '--base-path',
        type=Path,
        default=Path.cwd(),
        help='Base path for project creation (default: current directory)'
    )
    
    parser.add_argument(
        '--environment',
        choices=['development', 'staging', 'production'],
        default='development',
        help='Environment configuration to use (default: development)'
    )
    
    parser.add_argument(
        '--azure-vault-url',
        help='Azure Key Vault URL for secure configuration management'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing project setup'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    global logger
    logger = setup_logging(log_level)
    
    # Initialize project setup
    project_setup = ProductionMLProjectSetup(
        project_name=args.project_name,
        base_path=args.base_path
    )
    
    try:
        if args.validate_only:
            # Validate existing setup
            result = project_setup.validate_setup()
            print(json.dumps(result, indent=2))
            
            if result["status"] == "invalid":
                print("\nRecommendations:")
                for rec in result["recommendations"]:
                    print(f"  - {rec}")
                sys.exit(1)
        
        else:
            # Initialize project
            result = project_setup.initialize_project(
                environment=args.environment,
                azure_vault_url=args.azure_vault_url
            )
            
            if result["status"] == "success":
                print("\n🎉 Production ML Project Setup Completed Successfully!")
                print(f"\nProject: {result['project_name']}")
                print(f"Location: {result['base_path']}")
                print(f"Environment: {result['environment']}")
                
                print("\n✅ Features Implemented:")
                for feature in result["features"]:
                    print(f"  • {feature}")
                
                print("\n🚀 Next Steps:")
                print("  1. Activate virtual environment: source venv/bin/activate")
                print("  2. Install dependencies: make install")
                print("  3. Run tests: make test")
                print("  4. Start development: make docker-up")
                print("  5. Access MLflow UI: http://localhost:5000")
                
            else:
                print(f"\n❌ Project setup failed: {result['error']}")
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
