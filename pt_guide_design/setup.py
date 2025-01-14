from setuptools import setup, find_packages

setup(
    name="pt_guide_design",
    version="0.1.0",
    packages=find_packages(include=['pt_guide_design', 'pt_guide_design.*']),
    package_data={
        'pt_guide_design': ['resources/*', 'crisporWebsite/*']
    },
    install_requires=[
        "biopython==1.81",
        "gffutils==0.11.1",
        "pybedtools==0.8.0",
        "pandas==1.3.5",
        "numpy==1.21.6",
        "tqdm==4.65.0",
        "pyfaidx==0.7.1",
        "scikit-learn==1.0.2",
        "redis==3.5.3",
        "matplotlib==3.5.3"
    ],
    author="Jon Doenier",
    author_email="Doenierjon@gmail.com",
    description="CRISPR guide design for Phaeodactylum tricornutum",
    python_requires=">=3.7,<3.8",
) 