from setuptools import setup, find_packages

setup(
    name="IHSetBernabeu",
    version="0.1.0",
    description="Equilibrium beach profile, Bernabeu et al. (2003)",
    author="Lucas de Freitas Pereira",
    author_email="lucas.defreitas@unican.es",
    url="https://github.com/IHCantabria/IHSetBernabeu",
    license="MIT",

    # ### critical step for src layout ###
    package_dir={"": "src"},
    packages=find_packages(where="src"),

    include_package_data=True,
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.1",
        "matplotlib>=3.0",
        "scipy>=1.5",
        "xarray>=0.15",
        "numba>=0.50"
    ],

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],

    python_requires=">=3.7",
)
