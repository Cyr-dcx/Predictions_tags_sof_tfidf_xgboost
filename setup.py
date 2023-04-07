from setuptools import find_packages, setup

setup(
    name="fastAPI_predictions",
    packages=find_packages(
        include=["utils_package", "utils_package.*"]
    ),
    entry_points={
        "console_scripts": [
            "fastAPI_predictions=fastAPI_predictions.executor.cli:main",  # noqa
        ],
    },
    version="0.1.0",
    description="analyse d'un jeu de questions stackoverflow et de définition de tags",
    author="Cyr_Dcx"
)