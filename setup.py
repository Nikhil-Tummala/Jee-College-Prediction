"""
Setup script for JEE College Prediction project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="jee-college-predictor",
    version="1.0.0",
    author="Tummala Nikhil Phaneendra",
    author_email="tummala1911@gmail.com",
    description="A machine learning project to predict JEE college admission outcomes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nikhil-Tummala/Jee-College-Prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "jee-predictor=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.pkl", "*.joblib"],
    },
)
