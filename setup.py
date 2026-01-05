from setuptools import setup, find_packages

setup(
    name='alphaquarto-v1',
    version='0.1.0',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=['numpy', 'tensorflow'],
)
