import setuptools

setuptools.setup(
    name="piOED",
    version="0.1.0",
    author="Nicolai Palm",
    author_email="nicolaipalm@googlemail.com",
    description="Parameter individual optimal experimental design",
    packages=setuptools.find_packages(where="src"),
    url="https://github.com/nicolaipalm/oed",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    python_requires='>=3.6',
)
