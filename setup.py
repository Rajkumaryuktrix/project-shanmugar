from setuptools import setup, find_packages

setup(
    name="shanmugaa",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "colorama",
        "tabulate",
        "python-dotenv",
        "scipy"
    ],
    python_requires=">=3.8",
) 