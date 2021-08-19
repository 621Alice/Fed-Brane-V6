from os import path
from codecs import open
from setuptools import setup, find_packages

# we're using a README.md, if you do not have this in your folder, simply
# replace this with a string.
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='v6-fedavg-mnist',
    version="1.0.0",
    description='vantage6 FedAvg MNIST',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',#github url
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'vantage6-client',
        'numpy',
        'scikit-learn',
        'torch',
        'torchvision',
        'pandas'
        # list your dependancies here:
        # pandas, ...
    ]
)
