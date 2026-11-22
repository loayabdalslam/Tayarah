from setuptools import setup, find_packages

setup(
    name="the1",
    version="1.1.0",
    description="The1: Inductive Neuro-Symbolic World Model for Extreme Few-Shot Natural Law Discovery",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Loaii abdalslam",
    author_email="loaiabdalslam@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "sympy>=1.12",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "safetensors>=0.4.0",
        "huggingface_hub>=0.20.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
