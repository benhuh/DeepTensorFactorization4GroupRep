from setuptools import find_packages, setup

setup(
    name="dtf",
    packages=find_packages(),
    version="0.0.1",
    install_requires=[
        "pytorch_lightning",
        # "blobfile",
        "numpy",
        "torch",
        "tqdm",
        "scipy",
        "mod",
        "matplotlib",
        "jupyterlab",
        "tensorboard",
    ],
)
