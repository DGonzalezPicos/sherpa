from setuptools import setup, find_packages

setup(
    name="sherpa",
    version="0.1.0",
    description="SHERPA: Spectroscopic High-resolution Emission & Retrieval Pipeline for Atmospheres",
    author="Dario Gonzalez Picos",
    author_email="picos@strw.leidenuniv.nl",
    packages=find_packages(),
    package_data={"sherpa": ["data/chemistry/species_info.txt"]},
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "astropy",
        "pandas",
        "matplotlib",
        "corner",
        "petitradtrans>=3.0",
        "pymultinest",
        "PyAstronomy",
        "h5py",
        "pyfastchem",
        "spectres",
    ],
    python_requires=">=3.10",
)
