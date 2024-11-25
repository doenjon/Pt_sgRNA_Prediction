from setuptools import setup, find_packages

setup(
    name="sgrna_scorer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.9.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
    ],
    author="Jon Doenier",
    author_email="jdoenier@stanford.edu",
    description="sgRNA guide design for P. tricornutum",
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
)