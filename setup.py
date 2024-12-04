from setuptools import setup, find_packages

setup(
    name="neuron_network",
    version="0.1.0",
    description="Neuron Network",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Lucas Mialichi",
    author_email="lmcmialichi@gmail.com",
    url="https://github.com/seuusuario/my_package",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
