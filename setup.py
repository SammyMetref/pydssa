import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

def readme():
    with open('README.md') as f:
        return f.read()

setuptools.setup(
    name="or3d", # Replace with your own username
    version="0.0.1",
    author="Sammy Metref",
    author_email="sammy.metref@univ-grenoble-alpes.fr",
    description="Reconstruction of ocean 3d fields using surface maps and observations",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/SammyMetref/or3d",
    packages=['or3d'],
    intall_requires = [],
    include_package_data = True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
