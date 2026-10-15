from setuptools import setup, find_packages

setup(
    name="econocausal",
    version="0.1.0",
    description="EconoCausal: Double Machine Learning and Causal Inference for Discount Optimization",
    author="EconoCausal Team",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.9",
)
