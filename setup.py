from pathlib import Path

from setuptools import find_packages, setup


def read(rel_path):
    here = Path(__file__).parent.absolute()
    with open(here.joinpath(rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


# add the README.md file to the long_description
with open("README.md", "r") as fh:
    long_description = fh.read()


install_requires = [
    "flake8",
    "isort",
    "matplotlib",
    "opencv-python",
    "pandas>=2.0.0",
    "pytest",
    "scikit-learn",
    "seaborn",
    "typing",
]


setup(
    name="keypoint-pseudo-labeler",
    packages=find_packages(),
    version=get_version(Path("pseudo_labeler").joinpath("__init__.py")),
    description="Use EKS as a pseudo-labeler to accelerate pose estimation projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Paninski Lab",
    install_requires=install_requires,
    url="https://github.com/paninski-lab/keypoint-pseudo-labeler",
    keywords=["machine learning", "deep learning", "computer_vision"],
)
