from setuptools import setup, find_packages

setup(
    name="shanmugaa",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
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