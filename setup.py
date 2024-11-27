from setuptools import setup, find_packages

setup(
    name="sleep-quality-prediction",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pymc',
        'numpy',
        'pandas',
        'streamlit',
        'plotly',
        'pytest'
        'snet.sdk',
        'grpcio',
        'grpcio-tools',
        'protobuf'
    ],
)
