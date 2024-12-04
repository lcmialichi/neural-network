from setuptools import setup, find_packages

setup(
    name="neural_network",
    version="0.1.0",
    description="Neuron Network",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Lucas Mialichi",
    author_email="lmcmialichi@gmail.com",
    url="https://github.com/lcmialichi/neural-network",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas"
    ],
     entry_points={
        "console_scripts": [
            "neural-app=main:main",
        ]
     },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
