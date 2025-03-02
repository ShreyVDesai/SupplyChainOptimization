from setuptools import setup, find_packages

setup(
    name="SupplyChainOptimization",
    version="0.1.0",
    packages=find_packages(),
    description="Supply Chain Optimization Data Pipeline",
    author="Supply Chain Optimization Team",
    python_requires=">=3.10",
    # Read requirements from requirements.txt
    install_requires=[
        line.strip()
        for line in open("Data_Pipeline/requirements.txt").readlines()
        if not line.startswith("#") and line.strip() != ""
    ],
)
