from setuptools import setup, find_packages

setup(
    name="pt_guide_design",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "biopython",
        "gffutils",
        "pybedtools",
        "pandas",
        "numpy",
        "tqdm",
        "pyfaidx",
        "scikit-learn"
    ],
    author="Jon Doenier",
    author_email="Doenierjon@gmail.com",
    description="CRISPR guide design for Phaeodactylum tricornutum",
    python_requires=">=3.9",
) 