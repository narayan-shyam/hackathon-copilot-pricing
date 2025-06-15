from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="unified-dynamic-pricing",
    version="1.0.0",
    author="Dynamic Pricing Team",
    author_email="team@dynamicpricing.com",
    description="Unified Dynamic Pricing Pipeline - ML-powered pricing optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/unified-dynamic-pricing",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
    keywords="pricing, machine-learning, optimization, dynamic-pricing, ml-pipeline",
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=6.0.0", "pytest-cov", "black", "flake8"],
        "advanced": ["xgboost>=1.5.0", "lightgbm>=3.3.0", "statsmodels>=0.13.0"],
    },
    entry_points={
        "console_scripts": [
            "dynamic-pricing=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
