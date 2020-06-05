import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ising",
    version="0.0.1",
    author="Cameron Perot",
    description="A basic Ising model/Metropolis algorithm simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cameronperot/ising",
    packages=["ising"],
    package_dir={"ising": "src/ising"},
    classifiers=["Programming Language :: Python :: 3"],
    python_requires=">=3.6",
    install_requires=["numpy", "numba", "matplotlib"],
)
