from setuptools import setup, find_packages

setup(
    name="sgrna_scorer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow==2.10.1",
        "numpy==1.21.6",
        "pandas==1.3.5",
        "scikit-learn==1.0.2",
        "matplotlib==3.5.3"
    ],
    author="Jon Doenier",
    author_email="Doenierjon@gmail.com",
    description="Score sgRNA sequences for CRISPR activity",
    python_requires=">=3.7,<3.8",
)