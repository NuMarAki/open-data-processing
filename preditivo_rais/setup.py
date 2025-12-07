from setuptools import setup, find_packages

setup(
    name='preditivo_rais_cpu',
    version='0.1.0',
    description='Predictive modeling for RAIS using Random Forest on CPU.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'pyyaml',
        'scikit-learn',
        'joblib',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
