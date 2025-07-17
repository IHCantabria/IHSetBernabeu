from setuptools import setup, find_packages

setup(
    name='IHSetBernabeu',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'xarray',
        'numba'
    ],
    author='Lucas de Freitas Pereira',
    author_email='lucas.defreitas@unican.es',
    description='IH-SET Bernabeu et al. (2003)',
    url='https://github.com/IHCantabria/IHSetBernabeu',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
)
