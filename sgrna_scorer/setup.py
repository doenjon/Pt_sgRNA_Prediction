from setuptools import setup, find_packages

setup(
    name="sgrna-scorer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.0.0",
        "numpy>=1.19.0",
        "pandas>=1.0.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.0.0"
    ],
    include_package_data=True,
    package_data={
        'sgrna_scorer': [
            'resources/*.h5',
            'models/*.h5',
            'models/weights/*.h5'
        ]
    },
    author="Jon Doenier",
    author_email="Doenierjon@gmail.com",
    description="A package for predicting sgRNA efficiency scores in P. tricornutum",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)