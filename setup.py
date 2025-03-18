from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r") as req_file:
    requirements = [line.strip() for line in req_file if line.strip()]

setup(
    name="CryptoExplorer",
    version="1.0.3rc1",
    author="Archie Marques",
    description="Simplified cryptocurrency and blockchain data extraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
)
