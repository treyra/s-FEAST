"""
failurePy package
"""
from setuptools import setup, find_packages

#Add readme as long description
with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='failurePy',
    version='1.4.5',
    author='Jimmy Ragan',
    author_email='jragan@caltech.edu',
    description='Library of failure identification algorithms and models, estimators and scripts to test the algorithms against',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/treyra/s-FEAST/',
    project_urls = {
        "Bug Tracker": "https://github.com/treyra/s-FEAST/issues"
    },
    license='All Rights Reserved',
    packages=find_packages(),
    #Explicitly not putting HIL dependencies here, planning on separating these completely
    install_requires=['jax','numpy','matplotlib','pyyaml','tqdm','cvxpy','opencv-python'],
)
