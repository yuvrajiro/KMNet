from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kmnet",
    version="0.1.0",
    author="Yuvraj",
    author_email="yuvraj@gmail.com",
    description="A discrete-time survival analysis model with Kaplan-Meier inspired loss.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yuvrajiro/KMNet",
    project_urls={
        "Bug Tracker": "https://github.com/yuvrajiro/KMNet/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.7.0",
        "numpy",
        "pandas",
        "torchtuples",
        "pycox",
        "numba",
    ],
)
