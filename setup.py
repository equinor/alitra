from setuptools import find_packages, setup

setup(
    name="alitra",
    description="Simple alignment and transformation between coordinate frames",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    version="1.0.16",
    author="Equinor ASA",
    author_email="arnts@equinor.com, euel@equinor.com, chjo@equinor.com",
    license="MIT",
    url="https://github.com/equinor/alitra",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries",
    ],
    include_package_data=True,
    install_requires=["scipy", "numpy", "dacite"],
    python_requires=">=3.8",
    extras_require={
        "dev": [
            "pytest",
            "black",
        ]
    },
)
