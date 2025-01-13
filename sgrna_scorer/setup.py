from setuptools import setup, find_packages

setup(
    name="sgrna_scorer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.12.0",
        "numpy>=1.24.2",
        "scikit-learn>=1.2.2"
    ],
    include_package_data=True,
    package_data={
        'sgrna_scorer': ['resources/*.h5'],
    }
)