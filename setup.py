import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="astrorealm",
    version="0.0.1",
    author="Sthabile Kolwa",
    author_email="sthabile.kolwa@gmail.com",
    description='''This package provides modules for
    analysing the kinematics of gas in distant radio galaxies
    using telescope datasets.''',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thabsko/astrorealm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)